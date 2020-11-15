import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
from PIL import Image
import constant, json, random
from s2_model import s2_integrate

def get_image_clip(path, indices):
    clip = [path/(str(i).zfill(5)+".jpg") for i in indices]
    clip = np.array([np.asarray(Image.open(path)) for path in clip])
    clip = clip.transpose(0, 3, 1, 2)
    clip = np.float32(clip)

    return torch.from_numpy(clip)

def get_fov_embedding(path, user, video, indices, integrate_fov=True):
    embeddings = [path/'{}-{}-{}.pth'.format(user, video, i) for i in indices]
    if integrate_fov:
        embeddings = [s2_integrate(torch.load(x)) for x in embeddings]
    else:
        embeddings = [torch.load(x) for x in embeddings]
    return torch.stack(embeddings, 0)

def get_motion_embedding(path, user, video, start, end, lth):
    delay = constant.UPDATE_DELAY
    motion = torch.load(path/'{}-{}.pth'.format(user, video))
    embeddings = []
    for i in range(start, end):
        embedding = []
        for j in range(lth):
            idx = i-j+delay
            if idx < 0:
                idx = 0
            embedding = [motion[idx]]+embedding
        embedding = torch.cat(embedding, 0)
        embeddings.append(embedding)
    embeddings = torch.stack(embeddings, 0)
    return embeddings.type(torch.float32)

def get_video_embedding(path, video, indices, timewindow):
    delay = constant.UPDATE_DELAY
    embeddings = []
    for t in timewindow:
        embedding = [path/'{}-{}.pth'.format(video, i+t+delay) for i in indices]
        embedding = [torch.load(x).squeeze(0) for x in embedding]
        embedding = torch.stack(embedding, 0)
        embeddings.append(embedding)

    return torch.transpose(torch.stack(embeddings, 0), 0, 1)

def get_label_embedding(path, user, video, indices, timewindow):
    delay = constant.UPDATE_DELAY
    embeddings = []
    for t in timewindow:
        embedding = [path/'label-{}-{}-{}.pth'.format(user, video, i+t+delay) for i in indices]
        embedding = [torch.load(x) for x in embedding]
        embedding = torch.stack(embedding, 0)
        embeddings.append(embedding)

    return torch.transpose(torch.stack(embeddings, 0), 0, 1)

class s2_video_dataset(data.Dataset):
    def __init__(self, stage=0):
        self.root = Path(constant.FOV_ROOT_PATH)
        self.fov = self.root/'fov'
        self.label = self.root/'label'
        self.video = self.root/'video'
        self.motion = self.root/'motion'
        self.integrate_fov = constant.INTEGRATE_FOV
        
        self.seq_len = constant.FOV_SEQ_LEN
        self.timewindow = constant.PRED_TIME_WINDOW
        self.step = constant.FOV_STEP

        self.video_len = np.ceil((np.array(constant.FOV_VID_LENGTH)-self.seq_len-self.timewindow[-1]+1)/self.step).astype(int)
        # self.video_len = self.video_len.astype(int)
        self.pack_size = self.video_len.sum()

        if stage == 0:
            self.start_idx = constant.TRAIN_START_IDX
            self.total_size = self.pack_size*constant.TRAIN_DATA_SIZE
        elif stage == 1:
            self.start_idx = constant.VALID_START_IDX
            self.total_size = self.pack_size*constant.VALID_DATA_SIZE
        else:
            self.start_idx = constant.TEST_START_IDX
            self.total_size = self.pack_size*constant.TEST_DATA_SIZE
        
    def __getitem__(self, index):
        user_no = int(index/self.pack_size)+self.start_idx
        request_no = index%self.pack_size
        sum = 0
        
        for vid in range(self.video_len.size):
            if request_no < sum+self.video_len[vid]:
                fov, video, motion, label = self.prepare_data(user_no, vid+1, request_no-sum)
                idx = [user_no, vid+1, self.step*(request_no-sum)]
                break

            sum += self.video_len[vid]
            
        return fov, motion, video, label, torch.Tensor(idx)
    
    def __len__(self):
        return self.total_size
        
    def prepare_data(self, user_no, video_no, request_no):
        start = request_no * self.step
        
        indices = range(start, start+self.seq_len)
        fov = get_fov_embedding(self.fov, user_no, video_no, indices, self.integrate_fov)
        motion = get_motion_embedding(self.motion, user_no, video_no, start, start+self.seq_len, 3)

        video = get_video_embedding(self.video, video_no, indices, self.timewindow)
        label = get_label_embedding(self.label, user_no, video_no, indices, self.timewindow)
        
        return fov, video, motion, label

class encode_fov_embedding_dataset(data.Dataset):
    def __init__(self):
        self.fov = Path('/mnt/sdc1/fov_soft_240_embedding')

        self.video_len = np.array(constant.VID_LENGTH)
        self.pack_size = self.video_len.sum()
        self.total_size = self.pack_size*29
        self.start_idx = 11

    def __getitem__(self, index):
        user_no = int(index/self.pack_size)
        request_no = index%self.pack_size
        sum = 0
        for vid in range(self.video_len.size):
            if request_no < sum+self.video_len[vid]:
                embeddings = torch.load(self.fov/'{}-{}-{}.pth'.format(user_no+self.start_idx, vid+1, request_no-sum))
                return embeddings[::], torch.Tensor([user_no+self.start_idx, vid+1, request_no-sum])

            sum += self.video_len[vid]
            
    def __len__(self):
        return self.total_size

class encode_video_embedding_dataset(data.Dataset):
    def __init__(self):
        self.video = Path('/mnt/sdc1/video_soft_240_embedding')
        self.total_size = np.sum(constant.VID_LENGTH)
        self.video_len = constant.VID_LENGTH
        self.interval = constant.ENCODER_FRAME_INTERVAL

    def __getitem__(self, index):
        sum = 0
        for vid in range(18):
            if index < sum+self.video_len[vid]:
                encodings = torch.load(self.video/'{}-{}.pth'.format(vid+1, index-sum))
                return encodings[::self.interval], torch.Tensor([vid+1, index-sum])

            sum += self.video_len[vid]
            
    def __len__(self):
        return self.total_size

class backup_encode_fov_dataset(data.Dataset):
    def __init__(self):
        self.fov = Path('/mnt/sda1/wuchlei/fovsoft_240')
        self.video_len = np.array(constant.VID_LENGTH)
        self.pack_size = self.video_len.sum()
        self.total_size = self.pack_size*39

    def __getitem__(self, index):
        user_no = int(index/self.pack_size)
        request_no = index%self.pack_size
        sum = 0
        for vid in range(self.video_len.size):
            if request_no < sum+self.video_len[vid]:
                # embeddings = torch.load(self.fov/'{}-{}-{}.pth'.format(user_no, vid+1, start))
                # return embeddings[::], torch.Tensor([user_no, vid, start])
                request_no -= sum
                images = self.prepare_data(user_no+1, vid+1, request_no)
                return images, torch.Tensor([user_no+1, vid+1, request_no])

            sum += self.video_len[vid]
    
    def __len__(self):
        return self.total_size
    
    def prepare_data(self, user_no, video_no, start):
        fov_indices = range(start*16+1, start*16+17)
        fov_clip = get_image_clip(self.fov/str(user_no)/str(video_no), fov_indices)
        
        return fov_clip

  
class encode_fov_dataset(data.Dataset):
    def __init__(self):
        self.fov = Path('/mnt/sda1/wuchlei/fovsoft_240')
        self.total_size = 1

    def __getitem__(self, index):
        user_no = 11
        vid = 13
        request_no = 191
        
        images = self.prepare_data(user_no, vid, request_no)
        return images, torch.Tensor([user_no, vid, request_no])
    
    def __len__(self):
        return self.total_size
    
    def prepare_data(self, user_no, video_no, start):
        fov_indices = range(start*16+1, start*16+17)
        fov_clip = get_image_clip(self.fov/str(user_no)/str(video_no), fov_indices)
        
        return fov_clip

class encode_video_dataset(data.Dataset):
    def __init__(self):
        self.video = Path('/mnt/sdc1/video_soft_240')
        self.total_size = np.sum(constant.VID_LENGTH)
        self.video_len = constant.VID_LENGTH

    def __getitem__(self, index):
        sum = 0
        for vid in range(18):
            if index < sum+self.video_len[vid]:
                encodings  = self.prepare_data(vid+1, index-sum)
                return encodings, torch.Tensor([vid+1, index-sum])

            sum += self.video_len[vid]
            
    def __len__(self):
        return self.total_size
    
    def prepare_data(self, video_no, start):
        fov_indices = range(start*16+1, start*16+17)
        fov_clip = get_image_clip(self.video/str(video_no), fov_indices)
        return fov_clip


class simple_image_dataset(data.Dataset):
    def __init__(self):
        self.root = Path(constant.ROOT_PATH)
        self.fov = self.root/'fov'
        self.label = self.root/'label'
        self.frame = self.root/'video'
        self.interval = constant.ENCODER_FRAME_INTERVAL
        self.seq_len = constant.PREDICTOR_SEQ_LEN
        self.clas_res = json.load(open('data_class.json'))
        self.clas_no = [1000, 2000]
        self.total_size = np.sum(self.clas_no)


        self.data_pool = random.sample(self.clas_res['labels'][0], self.clas_no[0])+random.sample(self.clas_res['labels'][1], self.clas_no[1])
        self.label_name = "label-%d-%d-%d.pth"

    def refresh_pool(self):
        self.data_pool = random.sample(self.clas_res['labels'][0], self.clas_no[0])+random.sample(self.clas_res['labels'][1], self.clas_no[1])

    def __getitem__(self, index):
        user_no, vid, start = self.data_pool[index]
        encodings, targets = self.prepare_data(user_no, vid, start)
        return encodings, targets
    
    def __len__(self):
        return self.total_size
    
    def prepare_data(self, user_no, video_no, start):
        fov_indices = range(start*16+1, start*16+17, self.interval)
        fov_clip = get_image_clip(self.fov/str(user_no)/str(video_no), fov_indices)
        
        clips = [fov_clip]
        # targets = [torch.load(str(self.label/(self.label_name%(user_no, video_no, start))))]
        targets = [torch.load(str(self.label/(self.label_name%(user_no, video_no, start))))]

        for k in range(start+1, start+self.seq_len):
            video_indices = range(k*16+1, k*16+17, self.interval)
            video_clip = get_image_clip(self.frame/str(video_no), video_indices)
            clips.append(video_clip)
            
            targets.append(torch.load(str(self.label/(self.label_name%(user_no, video_no, k)))))

        inputs = torch.stack(clips,0)
        targets = torch.stack(targets,0)
        return inputs, targets

class image_dataset(data.Dataset):
    def __init__(self):
        self.root = Path(constant.ROOT_PATH)
        self.fov = self.root/'fov'
        self.label = self.root/'label'
        self.frame = self.root/'video'
        
        self.start_idx = constant.TRAIN_START_IDX
        self.seq_len = constant.PREDICTOR_SEQ_LEN
        self.step = constant.SAMPLE_TIMEWINDOW
        self.interval = constant.ENCODER_FRAME_INTERVAL

        self.video_len = (np.array(constant.VID_LENGTH)-self.seq_len+1)/self.step
        self.video_len = self.video_len.astype(int)
        self.pack_size = self.video_len.sum()

        self.total_size = self.pack_size*constant.TRAIN_DATA_SIZE
        
        self.label_name = "label-%d-%d-%d.pth"

    def __getitem__(self, index):
        user_no = int(index/self.pack_size)+self.start_idx
        request_no = index%self.pack_size
        sum = 0
        for vid in range(self.video_len.size):
            if request_no < sum+self.video_len[vid]:
                encodings, targets = self.prepare_data(user_no, vid+1, request_no-sum)
                break
            sum += self.video_len[vid]
            
        return encodings, targets
    
    def __len__(self):
        return self.total_size

        
    def prepare_data(self, user_no, video_no, request_no):
        start = request_no * self.step
        
        fov_indices = range(start*16+1, start*16+17, self.interval)
        fov_clip = get_image_clip(self.fov/str(user_no)/str(video_no), fov_indices)
        
        clips = [fov_clip]
        targets = []

        for k in range(start+1, start+self.seq_len):
            video_indices = range(k*16+1, k*16+17, self.interval)
            video_clip = get_image_clip(self.frame/str(video_no), video_indices)
            clips.append(video_clip)
            
            targets.append(torch.load(str(self.label/(self.label_name%(user_no, video_no, k)))))

        inputs = torch.stack(clips,0)
        targets = torch.stack(targets,0)
        return inputs, targets
