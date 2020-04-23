import os


class Writer:
    def __init__(self, output_dir='./output/', file='log'):
        self.output_dir = output_dir
        self.file = file
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                raise

    def w(self, data):
        with open(f'{self.output_dir}/{self.file}', 'a') as f:
            f.write('----------------------------------------------\n')
            if isinstance(data, str):
                f.write(data+'\n')
            if isinstance(data, list):
                data = [str(_) for _ in data]
                f.write(' '.join(data)+'\n')
            if isinstance(data, dict):
                for k, v in data.items():
                    f.write(f'{str(k)} {str(v)}\n')