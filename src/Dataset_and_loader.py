import torch
from torch.utils.data import Dataset, DataLoader

class Implement_Dataset(Dataset):
    def __init__(self, text, max_length, stride):
        self.input_id = []
        self.target_id = []

        tokenizer = GPT_Tokenizer()
        token_id = tokenizer.encode(text)

        # range should use step=stride, not a second argument
        for i in range(0, len(token_id) - max_length, stride):
            input_seq = token_id[i : i + max_length]
            target_seq = token_id[i + 1 : i + max_length + 1]

            self.input_id.append(input_seq)
            self.target_id.append(target_seq)

    def __len__(self):
        return len(self.input_id)

    def __getitem__(self, idx):
        return torch.tensor(self.input_id[idx]), torch.tensor(self.target_id[idx])

# class Implement_Dataset(Dataset):
#    def __init__(self, text, max_length , stride):
#      input_id = []
#      target_id = []

#      tokenizer = tokenizer = GPT_Tokenizer()
#      token_id = tokenizer.encode(text)

#      for i in range(len(token_id)- max_length, stride):
#         input  = token_id[i : i+max_length]
#         target = token_id[i+1 : i+max_length+1]

#         input_id.append(input)
#         target_id.append(target)

#         self.input_id = input_id
#         self.target_id = target_id

#    def __len__(self):
#        return len(self.input_id)


#    def __getitem__(self, idx):
#       return torch.tensor(self.input_id[idx]), torch.tensor(self.target_id[idx])

"""This code implements a custom dataset class for our LLM training. Next, we’ll create a DataLoader to fetch the data in batches and iterate over it during training."""

def Implement_DataLoader(txt,batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):
      dataset = Implement_Dataset(txt, max_length, stride)
      dataloader = DataLoader(dataset,
                              batch_size= batch_size ,
                              shuffle= shuffle,
                              drop_last= drop_last ,
                              num_workers=num_workers)
      return dataloader

dataloader = Implement_DataLoader(text_data, batch_size=4, max_length=8, stride=4)

data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)