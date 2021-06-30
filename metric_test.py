# 用于EE任务的离线评测
import json
import codecs
from TextMetric import ACE05TextMetric, DuEETextMetric, DuEEFinTextMetric
import argparse
# argument 去重在读数据时候进行处理
def dueeResultRead(path):

    id_result = {} # {id: [pre_spans]}

    with codecs.open(path, 'r', 'UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            role_set = set() # argument 去重
            
            text_id = line["id"]
            span_list = []
            for e in line["event_list"]:
                et = e["event_type"]
                for r in e["arguments"]:
                    rt = r["role"]
                    argument = r["argument"]

                    # span_list.append((argument, et + "-" + rt))
                    
                    if et + "-" + rt + "-" + argument not in role_set:
                        span_list.append((argument, et + "-" + rt))
                        role_set.add(et + "-" + rt + "-" + argument)
                    
            id_result[text_id] = span_list


    return id_result

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--method",
    #                     type=str,
    #                     required=True,
    #                     help="choose TextMetric")
    pre_path = "data/DuEE/duee_bert_base_dev.json"
    gold_path = "data/DuEE/dev.json"
    
    pre_result = dueeResultRead(pre_path)
    gold_result = dueeResultRead(gold_path)

    pre_spans = []
    gold_spans = []
    
    for tid, spans in gold_result.items():
        if tid in pre_result:
            pre_spans.append(pre_result[tid])
            gold_spans.append(spans)
            # print("pre:", pre_result[tid])
            # print("gold:", spans)
        else:
            pre_spans.append([])
            gold_spans.append(spans)
    
    # ace05 metric 测试
    # ace05metric = ACE05TextMetric()
    # ace05metric(pre_spans, gold_spans) # 调用__call__函数
    #
    # i_p, i_r, i_f, c_p, c_r, c_f = ace05metric.get_metric()
    #
    #
    # print({"r_i_p": i_p, "r_i_r": i_r, "r_i_f": i_f,
    #             "r_c_p": c_p, "r_c_r": c_r, "r_c_f": c_f})

    dueemetric = DuEEFinTextMetric()
    dueemetric(pre_spans, gold_spans)
    print(gold_spans)
    precision, recall, f1_score = dueemetric.get_metric()
    print(precision, recall, f1_score)