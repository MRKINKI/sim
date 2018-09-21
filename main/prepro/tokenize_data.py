import json


class TokenizeData:
    def __int__(self):
        pass

    def run(self, input_file, output_file, tokenizer):
        with open(output_file, 'w', encoding='utf-8') as fout:
            with open(input_file, encoding='utf-8') as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    sample['text_words'] = tokenizer.segment(sample['text'])
                    fout.write(json.dumps(sample)+'\n')
