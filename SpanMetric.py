# coding: UTF-8
# 通过元素在原文中的位置进行metirc评估, 即通过start、end是否一致来计算 p, r, f

import torch
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Any

# span metric 时候用到，即通过start/end来进行指标的评估
class CustomSpan():
    def __init__(self,
                 span_start: int,
                 span_end: int,
                 span_label: int,
                 extra_id: int) -> None:
        self.span_start = span_start
        self.span_end = span_end
        self.span_label = span_label
        self.extra_id = extra_id

        if not isinstance(span_start, int) or not isinstance(span_end, int):
            raise TypeError(f"SpanFields must be passed integer indices. Found span indices: "
                            f"({span_start}, {span_end}) with types "
                            f"({type(span_start)} {type(span_end)})")
        if span_start > span_end:
            raise ValueError(f"span_start must be less than span_end, "
                             f"but found ({span_start}, {span_end}).")


    def empty_field(self):
        return CustomSpan(-1, -1, -1, -1)

    def __str__(self) -> str:
        return f"SpanField with spans: ({self.span_start}, {self.span_end}, {self.span_label}, {self.extra_id})."

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 4:
            return other == (self.span_start, self.span_end, self.span_label, self.extra_id)
        else:
            return id(self) == id(other)
    
    def get_tuple_span(self) -> tuple:
        return (self.span_start, self.span_end, self.span_label, self.extra_id)


class SpanMetric():
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

        tuple: (start, end, label)
        predictions : list of list(tuple), required. eg: [[(0,1,1), (0,0,0)], [(0,0,0)]]  sentence_num * span_num * tuple
            The result of predictions.
        gold_labels : list of list(tuple), required. eg: [[(1,1,1), (1,1,1)], [(0,1,1)]]
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


# ACE05学术评测指标计算, 针对start/end位置进行评测打分
class ACE05SpanMetric(SpanMetric):
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
                    if (g_span[0] == p_span[0]) and (g_span[1] == p_span[1]): # start, end 起始位置准确
                        gold_span_i_flag[i] = 1
                        pred_span_i_flag[j] = 1

                    if (g_span[0] == p_span[0]) and (g_span[1] == p_span[1]) and (g_span[2] == p_span[2]): # start, end 起始位置准确且分类标签准确
                        gold_span_c_flag[i] = 1
                        pred_span_c_flag[j] = 1

            i_g_tp = sum(gold_span_i_flag)
            i_p_tp = sum(pred_span_i_flag)

            i_fn = gold_span_num - i_g_tp
            i_fp = pred_span_num - i_p_tp

            c_tp = sum(gold_span_c_flag)
            c_fn = gold_span_num - c_tp
            c_fp = pred_span_num - c_tp

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
class DuEESpanMetric(SpanMetric):
    def __init__(self):
        raise NotImplementedError
    
    def __call__(self, pred_spans, gold_spans):
        raise NotImplementedError
    
    def get_metric(self, reset=False):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

# 百度篇章级事件抽取评估指标
class DuEEFinSpanMetric(SpanMetric):
    def __init__(self):
        raise NotImplementedError
    
    def __call__(self, pred_spans, gold_spans):
        raise NotImplementedError
    
    def get_metric(self, reset=False):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

# CCKS21 金融篇章级抽取方案评测 https://www.biendata.xyz/competition/ccks_2021_task6_1/evaluation/
class CCKSFinSpanMetric(SpanMetric):
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
    ace05metric = ACE05SpanMetric()    
    pre_spans = [[(1, 1, 1)] * 10]
    gold_spans = [[(1, 1, 1)] * 10]
    ace05metric(pre_spans, gold_spans)
    i_p, i_r, i_f, c_p, c_r, c_f = ace05metric.get_metric()
    print({"t_i_p": i_p, "t_i_r": i_r, "t_i_f": i_f,
                "t_c_p": c_p, "t_c_r": c_r, "t_c_f": c_f})




