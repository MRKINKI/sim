import json


class PreproData:
    def __int__(self, tokenizer):
        self.tokenizer = tokenizer

    def run(self, input_file, output_file, fields):
        with open(output_file, 'w', encoding='utf-8') as fout:
            with open(input_file, encoding='utf-8') as fin:
                for line in fin:
                    sample = json.loads(line.strip())
                    for field in fields:
                        sample[field] = self.tokenizer.segment(sample[field])
                    fout.write(json.dumps(sample)+'\n')
