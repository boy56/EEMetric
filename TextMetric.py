# 基于元素内容进行metric评估

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Iterable, Optional, Any
from collections import defaultdict

class TextMetric():
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """

    supports_distributed = False

    def __call__(
        self, predictions, gold_labels
    ):
        """
        # Parameters

        tuple: (str, label)
        predictions : list of list(tuple), required. eg: [[("阿里",1), ("字节",0)], [("华为",0)]]  sentence_num * span_num * tuple
            The result of predictions.
        gold_labels : list of list(tuple), required. eg: [[("阿里",1), ("字节",1)], [("百度",1)]]
            The gold label.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError


# ACE05学术评测指标计算
class ACE05TextMetric(TextMetric):
    def __init__(self):
        self.i_g_tp = self.i_p_tp = 0.
        self.c_tp = 0.
        self.i_fp = self.c_fp = 0.
        self.i_fn = self.c_fn = 0.

    def __call__(self, pred_spans, gold_spans):
        assert len(pred_spans) == len(gold_spans)
        if isinstance(gold_spans, torch.Tensor):
            gold_spans = gold_spans.cpu().numpy().tolist()
        batch_size = len(pred_spans)
        # print(batch_size)
        # print(gold_spans)
        # 针对多个句子的span进行处理, 句子 * 句子长度个custom span
        for idx in range(batch_size):
            sentence_pred_span = pred_spans[idx]
            sentence_gold_span = gold_spans[idx]

            gold_span_i_flag = [0 for i in range(len(sentence_gold_span))]
            gold_span_c_flag = [0 for i in range(len(sentence_gold_span))]

            pred_span_i_flag = [0 for i in range(len(sentence_pred_span))]
            pred_span_c_flag = [0 for i in range(len(sentence_pred_span))]

            pred_span_num = len(sentence_pred_span)
            gold_span_num = 0
            # print(pred_span_num)
            for i, g_span in enumerate(sentence_gold_span):
                # print(g_span)
                if g_span[0] == -1:
                    break
                gold_span_num += 1
                for j, p_span in enumerate(sentence_pred_span):
                    if (g_span[0] == p_span[0]): # str 是否准确
                        gold_span_i_flag[i] = 1
                        pred_span_i_flag[j] = 1
                        

                    if (g_span[0] == p_span[0]) and (g_span[1] == p_span[1]): # str 准确且 label 准确
                        gold_span_c_flag[i] = 1
                        pred_span_c_flag[j] = 1

            
            i_g_tp = sum(gold_span_i_flag)
            i_p_tp = sum(pred_span_i_flag)
            

            # 因为sum(gold_span_i_flag)在某些情况下不等于sum(pred_span_i_flag), 因此需要分开来计算 (只用元素来判断是否识别出来, 因此会有某些重复结果导致两者不一致) 
            i_fn = gold_span_num - i_g_tp
            i_fp = pred_span_num - i_p_tp

            
            c_tp = sum(gold_span_c_flag)
            c_fn = gold_span_num - c_tp
            c_fp = pred_span_num - c_tp

            '''
            # 因为sum(gold_span_c_flag) = sum(pred_span_c_flag), 因此用一个c_tp就可以
            if sum(gold_span_c_flag) != sum(pred_span_c_flag):
                print(gold_span_c_flag, pred_span_c_flag)
                break
            
            '''

            self.i_g_tp += i_g_tp
            self.i_p_tp += i_p_tp
            self.i_fp += i_fp
            self.i_fn += i_fn
            self.c_tp += c_tp
            self.c_fp += c_fp
            self.c_fn += c_fn

    def _get_p_r_f(self, tp, fp, fn):
        p = float(tp) / float(tp + fp + 1e-13)
        r = float(tp) / float(tp + fn + 1e-13)
        f = 2. * ((p * r) / (p + r + 1e-13))
        return p, r, f

    def _get_i_p_r_f(self, g_tp, p_tp, fp, fn):
        p = float(p_tp) / float(p_tp + fp + 1e-13)
        r = float(g_tp) / float(g_tp + fn + 1e-13)
        f = 2. * ((p * r) / (p + r + 1e-13))
        return p, r, f

    def get_metric(self, reset=False):
        i_p, i_r, i_f = self._get_i_p_r_f(self.i_g_tp, self.i_p_tp, self.i_fp, self.i_fn)
        c_p, c_r, c_f = self._get_p_r_f(self.c_tp, self.c_fp, self.c_fn)
        if reset:
            self.reset()
        return i_p, i_r, i_f, c_p, c_r, c_f

    def reset(self):
        self.i_g_tp = self.i_p_tp = 0.
        self.c_tp = 0.
        self.i_fp = self.c_fp = 0.
        self.i_fn = self.c_fn = 0.


# 百度句级事件抽取评估指标
class DuEETextMetric(TextMetric):
    def __init__(self):
        self.score = 0.
        self.gold_num = 0
        self.pred_num = 0

    def __call__(self, pred_spans, gold_spans):
        assert len(pred_spans) == len(gold_spans)
        if isinstance(gold_spans, torch.Tensor):
            gold_spans = gold_spans.cpu().numpy().tolist()
        score, gold_num, pred_num = self.call_metric(pred_spans, gold_spans)
        self.score = score
        self.gold_num = gold_num
        self.pred_num = pred_num

    def call_metric(self, pred_spans, gold_spans):
        batch_size = len(pred_spans)
        score = 0.
        gold_arg_num = 0
        pred_arg_num = 0
        for idx in range(batch_size):
            sentence_pred_span = pred_spans[idx]
            sentence_gold_span = gold_spans[idx]
            gold_arg_num += len(sentence_gold_span)
            pred_arg_num += len(sentence_pred_span)
            for i, p_span in enumerate(sentence_pred_span):
                if i == -1:
                    break
                correct_len = 0
                pred_len = len(p_span[0])
                gold_len = 0
                for j, g_span in enumerate(sentence_gold_span):
                    if g_span[1] != p_span[1]:
                        continue
                    inter_len = len(set(p_span[0]) & set(g_span[0]))
                    if inter_len > correct_len:
                        gold_len = len(g_span[0])
                        correct_len = inter_len
                token_p = float(correct_len) / float(pred_len + 1e-13)
                token_r = float(correct_len) / float(gold_len + 1e-13)
                token_f1 = 2 * token_p * token_r / (token_p + token_r + 1e-13)
                score += token_f1
        return score, gold_arg_num, pred_arg_num

    
    def get_metric(self, reset=False):
        precision = float(self.score / self.pred_num + 1e-13)
        recall = float(self.score / self.gold_num + 1e-13)
        f1_score = float(2 * precision * recall / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_score
    
    def reset(self):
        self.score = 0.
        self.gold_num = 0
        self.pred_num = 0

# 百度篇章级事件抽取评估指标
class DuEEFinTextMetric(TextMetric):
    def __init__(self):
        self.correct_num = 0
        self.gold_num = 0
        self.pred_num = 0
    
    def __call__(self, pred_spans, gold_spans):
        batch_size = len(pred_spans)
        correct_num = 0
        gold_num = 0
        pred_num = 0
        for idx in range(batch_size):
            sentence_pred_span = pred_spans[idx]
            sentence_gold_span = gold_spans[idx]
            gold_num += len(sentence_gold_span)
            pred_num += len(sentence_pred_span)
            for i, p_span in enumerate(sentence_pred_span):
                if i == -1:
                    break
                for j, g_span in enumerate(sentence_gold_span):
                    if g_span[0] == p_span[0] and g_span[1] == p_span[1]:
                        correct_num += 1
                        break
            self.correct_num = correct_num
            self.gold_num = gold_num
            self.pred_num = pred_num
    
    def get_metric(self, reset=False):
        precision = float(self.correct_num / self.pred_num + 1e-13)
        recall = float(self.correct_num / self.gold_num + 1e-13)
        f1_score = float(2 * precision * recall / (precision + recall + 1e-13))
        if reset:
            self.reset()
        return precision, recall, f1_score
    
    def reset(self):
        self.correct_num = 0
        self.gold_num = 0
        self.pred_num = 0

# CCKS21 金融篇章级抽取方案评测 https://www.biendata.xyz/competition/ccks_2021_task6_1/evaluation/
class CCKSFinTextMetric(TextMetric):
    def __init__(self):
        raise NotImplementedError
    
    def __call__(self, pred_spans, gold_spans):
        raise NotImplementedError
    
    def get_metric(self, reset=False):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

# CCKS21 面向通信领域的过程类知识抽取 https://www.biendata.xyz/competition/ccks_2021_cpe_1/evaluation/



if __name__ == '__main__':
    
    # ace05 metric 测试
    ace05metric = ACE05TextMetric()    
    pre_spans = [[("阿里", 1)] * 10]
    gold_spans = [[("阿里", 1)] * 10]
    
    ace05metric(pre_spans, gold_spans) # 调用__call__函数
    i_p, i_r, i_f, c_p, c_r, c_f = ace05metric.get_metric()
    
    
    print({"t_i_p": i_p, "t_i_r": i_r, "t_i_f": i_f,
                "t_c_p": c_p, "t_c_r": c_r, "t_c_f": c_f})




