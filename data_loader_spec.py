from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import boto3
import s3tree
import numpy as np
import time

s3_client = boto3.client('s3')
s3tree.config.aws_access_key_id = 'AKIAJFZU4VXKST6P5BSA'
s3tree.config.aws_secret_access_key = '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'

class Spectrograms(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.mode = mode
        self.train_dataset = []
        self.preprocess()
        self.num_images = len(self.train_dataset)
        self.shuffled = np.arange(self.num_images)
        np.random.shuffle(self.shuffled)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        num_embeds=10
        embeddings = np.load('../embeddings.npy', allow_pickle=True)
        ids = np.load('../ids.npy', allow_pickle=True)
        speaker2idx = dict()
        for idx in range(len(ids)):
            if ids[idx][2:] in speaker2idx:
                speaker2idx[ids[idx][2:]].append(idx)
            else:
                speaker2idx[ids[idx][2:]] = [idx]
        tree = s3tree.S3Tree(bucket_name = 'face2speech', path='voxceleb/dev/aac')
        count =0 
        for folder in tree.directories:
            start = time.time()
            print(folder)
            voices = []
            for video in folder.get_tree():
                for file in video.get_tree().files:
                    if file.name.endswith('.npy'):
                        voices.append(file.path)
            for voice in voices:
                voice_id = folder.name[2:]
                embed_id = voice_id
                # print(voice_id)
                if folder.name[2:] in speaker2idx:
                    ems= speaker2idx[folder.name[2:]]
                    # count1=0

                    for em in ems[0:num_embeds]:
                        embedding = embeddings[em]
                        # if count1 < 10:
                        self.train_dataset.append([voice_id, embed_id, voice, embedding])
                        # else: 
                        #     break
                        
                        # print (voice_id, voice_id)
            count+=1
            print ("Time taken is {}".format(time.time() - start))


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset 

        if self.mode == "test":
            voiceid, _, file, _ = dataset[index]
            _, embedId, _, label = dataset[self.shuffled[index]]
        else:
            voiceid, embedId, file, label = dataset[index]

        spec = s3_client.download_file(Bucket='face2speech', Key=file, Filename='temp.jpeg')
        if spec.shape[1] < 300:
            spec = np.pad(spec, 300-spec.shape[1], 'wrap') 
        elif spec.shape[1] > 300:
            spec = spec[:,:300]

        assert(spec.shape[1] == 300)
              
        return voiceid, embedId, torch.FloatTensor(spec), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_size=256, batch_size=8, mode='train', num_workers=4):
    """Build and return a data loader."""
    
    if mode == "test":
        batch_size = 1
    spec_dataset = Spectrograms(mode)
    data_loader = data.DataLoader(dataset=spec_dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == "train"),
                                  num_workers=num_workers)
    return data_loader