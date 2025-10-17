import os, json
import argparse

# files = os.listdir('results')
# for file in files:
    # filename = os.path.join('results', file)
    # compensated = False
def score(filename: str, compensated: bool = False):
    # output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
    output = ["\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    # name = '.'.join(file.split('.')[:-1])
    # output.append(name+'\t'+str(round(100*(easy_acc+hard_acc)/len(pred_data), 1))+'\t'+str(round(100*easy_acc/easy, 1))+'\t'+str(round(100*hard_acc/hard, 1))+'\t'+str(round(100*short_acc/short, 1))+'\t'+str(round(100*medium_acc/medium, 1))+'\t'+str(round(100*long_acc/long, 1)))
    output.append('samples'+'\t'+str(len(pred_data))+'\t'+str(easy)+'\t'+str(hard)+'\t'+str(short)+'\t'+str(medium)+'\t'+str(long))
    output.append('acc'+'\t'+str(round(100*(easy_acc+hard_acc)/max(1,len(pred_data)), 1))+'\t'+str(round(100*easy_acc/max(1, easy), 1))+'\t'+str(round(100*hard_acc/max(1, hard), 1))+'\t'+str(round(100*short_acc/max(1, short), 1))+'\t'+str(round(100*medium_acc/max(1, medium), 1))+'\t'+str(round(100*long_acc/max(1, long), 1)))
    print('\n'.join(output))

# open('result.txt', 'w', encoding='utf-8').write('\n'.join(output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--compensated", "-c", action="store_true")
    args = parser.parse_args()
    score(args.file, args.compensated)
