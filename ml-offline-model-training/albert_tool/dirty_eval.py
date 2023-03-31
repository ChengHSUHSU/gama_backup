

import numpy as np
import pandas as pd
from albert_tool.eval import ndcg_score



def evaluation(hot_dat, popu_dat, gt_dat):
    # hot_dat->
    # content_id, final_score
    hot_dat = hot_dat.sort_values(by='final_score', ascending=False)
    hot_pred = list(hot_dat['content_id'])
    hot_pred2score = dict()
    for record in hot_dat.to_dict('records'):
        hot_pred2score[record['content_id']] = record['final_score']
    # popu_dat->
    # uuid, final_score
    popu_dat = popu_dat.sort_values(by='final_score', ascending=False)
    popu_pred = list(popu_dat['uuid'])
    popu_pred2score = dict()
    for record in popu_dat.to_dict('records'):
        popu_pred2score[record['uuid']] = record['final_score']
    # gt_dat->
    # userid, content_id, date, hour
    gt_content_id = set(gt_dat['content_id'])

    gt_dat = gt_dat[gt_dat['event'].isin(['home_page_content_click', 'planet_content_click', 'exploration_page_content_click'])]
    print('gt_dat : ', gt_dat.shape)
    print('gt_content_id : ', len(gt_content_id))

    print('====V===')
    print('hot_pred : ', len(hot_pred))
    print('popu_pred : ', len(popu_pred))
    print('====V===')
    hot_pred = list(set(hot_pred) & set(gt_content_id))
    popu_pred = list(set(popu_pred) & gt_content_id)
    print('hot_pred : ', len(hot_pred))
    print('popu_pred : ', len(popu_pred))
    
    #gt_dat = gt_dat[gt_dat['content_id'].isin(hot_pred)]
    gt_dat = gt_dat.sort_values(by='date')
    gt_dat = gt_dat.sort_values(by='hour')
    gt_dat = gt_dat.sort_values(by='userid', ascending=False)
    gt_dat = gt_dat[['userid', 'content_id']].drop_duplicates()
    print(gt_dat.shape)

    import random
    from tqdm import tqdm
    user_list = random.sample(list(set(gt_dat['userid'])), len(list(set(gt_dat['userid']))))

    for k in [5, 10, 15, 20]:
        print('TOPK : ', k)
        ndcg_k_hot, ndcg_k_popu = [], [] 
        for u in tqdm(user_list):
            dat_  = gt_dat[gt_dat['userid']==u]
            true_content_id = list(set(dat_['content_id']) )#[5,9,1,11,...]
            ##
            try:
                if len(true_content_id) >= 1:
                    # HOT
                    y_true, y_score = [], []
                    for id_ in hot_pred:
                        if id_ in true_content_id:
                            y_true.append(1)
                        else:
                            y_true.append(0.0001)
                        if id_ in hot_pred2score:
                            y_score.append(hot_pred2score[id_])
                        else:
                            y_score.append(0.0001)
                    ndcg_ = ndcg_score(y_true=y_true, y_score=y_score, k=k)
                    if pd.isna(ndcg_) is False:
                        ndcg_k_hot.append(ndcg_)
                    ## POPU
                    y_true, y_score = [], []
                    for id_ in popu_pred:
                        if id_ in true_content_id:
                            y_true.append(1)
                        else:
                            y_true.append(0.0001)
                        if id_ in popu_pred2score:
                            y_score.append(popu_pred2score[id_])
                        else:
                            y_score.append(0.0001)
                    # NDCG
                    ndcg_ = ndcg_score(y_true=y_true, y_score=y_score, k=k)
                    if pd.isna(ndcg_) is False:
                        ndcg_k_popu.append(ndcg_)
            except:
                pass
        if len(ndcg_k_hot)!=0:
            ndcg_k_hot = sum(ndcg_k_hot) / len(ndcg_k_hot)
        if len(ndcg_k_popu)!=0:
            ndcg_k_popu = sum(ndcg_k_popu) / len(ndcg_k_popu)
        print('ndcg_k_hot : ', ndcg_k_hot)
        print('ndcg_k_popu : ', ndcg_k_popu)
        print('====================')
    return



    pd.set_option("display.max_columns",1000000000)
    print('popularity_news : ', popularity_news.head(3))
    print('user_event : ', user_event.head(3))

pd.set_option("display.max_columns",1000000000)