import numpy as np
import time
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist

import utils


@torch.no_grad()
def evaluation(model, data_loader, device, args):
    # test
    model.eval()

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = data_loader.dataset.preprocess_text(text).to(device)
        text_embed = model.encode_text(text_input)
        text_embeds.append(text_embed)

    text_embeds = torch.cat(text_embeds, dim=0)

    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed = model.encode_image(image)
        image_embeds.append(image_embed)

    image_embeds = torch.cat(image_embeds, dim=0)

    score_matrix_i2t, score_matrix_t2i = model.get_similarity(
        image_embeds, text_embeds)
    score_matrix_i2t = score_matrix_i2t.contiguous()
    score_matrix_t2i = score_matrix_t2i.contiguous()
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    reserveNumber = 2

    eval_result = {'txt_r1': round(tr1,reserveNumber),
                   'txt_r5': round(tr5,reserveNumber),
                   'txt_r10': round(tr10,reserveNumber),
                   'txt_r_mean': round(tr_mean,reserveNumber),
                   'img_r1': round(ir1,reserveNumber),
                   'img_r5': round(ir5,reserveNumber),
                   'img_r10': round(ir10,reserveNumber),
                   'img_r_mean': round(ir_mean,reserveNumber),
                   'r_mean': round(r_mean,reserveNumber)}
    return eval_result
