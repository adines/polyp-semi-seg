import gc
import glob
import cv2
from tqdm import tqdm
from pathlib import Path
import torch
import PIL
from torchvision import transforms
from utils import *

from fastai.vision.all import *
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.all import *


def omniData(path, model,backbone, transformations, size=(320,320)):
    dls = get_dls(path, size, bs=1)
    nClasses = 2
    learn = getLearner(model, backbone, nClasses, path, dls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learn.load(model + '_' + backbone,device=device,with_opt=False)
    learn.model.to(device)

    del dls
    gc.collect()
    torch.cuda.empty_cache()

    images = sorted(glob.glob(path+os.sep+'unlabeled_images' + os.sep + "*"))
    newPath = path + "_tmp"
    if not os.path.exists(newPath):
        shutil.copytree(path, newPath)
    else:
        raise Exception("The path " + newPath + " already exists")

    print('Processing images with ' + model + ' model')
    for image in tqdm(images):
        name = image.split(os.sep)[-1]
        img = PIL.Image.open(image)
        imag = transforms.Resize(size)(img)
        tensor = transformImage(image=imag)

        lista = []

        pn = learn.model(tensor)
        lista.append(pn.cpu().detach())

        im = cv2.imread(image, 1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for transform in transformations:
            img = getTransform(transform, im)
            img = PIL.Image.fromarray(img)
            imag = transforms.Resize(size)(img)
            tensor = transformImage(image=imag)
            p = learn.model(tensor)
            p = np.asarray(p.cpu().detach())
            p=p[0]
            axis0=p.shape[0]
            for i in range(axis0):
                if i==0:
                    res=np.expand_dims(getTransformReverse(transform, p[i]),axis=0)
                else:
                    res=np.append(res, np.expand_dims(getTransformReverse(transform, p[i]),axis=0), axis=0)
            p=np.expand_dims(res, axis=0)
            lista.append(torch.from_numpy(p).cpu().detach())
            gc.collect()
            torch.cuda.empty_cache()
        prob, indices = averageVotingEnsemble(lista)
        newMask = getImageFromOut(indices,size,path + os.sep + 'codes.txt')

        img.save(newPath + os.sep + 'Images' + os.sep + 'train' + os.sep + 'new_' + name)
        newMask.save(newPath + os.sep + 'Labels' + os.sep + 'train' + os.sep + 'new_' + name)
        gc.collect()
        torch.cuda.empty_cache()
    del learn
    gc.collect()
    torch.cuda.empty_cache()


def omniModel(path,models,backbones,size=(320,320)):
    images = sorted(glob.glob(path+os.sep+'unlabeled_images' + os.sep + "*"))
    newPath = path + "_tmp"
    if not os.path.exists(newPath):
        shutil.copytree(path, newPath)
    else:
        raise Exception("The path " + newPath + " already exists")

    predictions = {}
    nClasses = 2
    for model, backbone in zip(models,backbones):
        dls = get_dls(path, size, bs=1)
        learn = getLearner(model, backbone, nClasses, path, dls)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learn.load(model + '_' + backbone,device=device,with_opt=False)
        learn.model.to(device)
        del dls
        gc.collect()
        torch.cuda.empty_cache()

        print('Processing images with ' + model + ' model')
        for image in tqdm(images):
            name = image.split(os.sep)[-1]
            if name not in predictions:
                predictions[name]=[]
            img = PIL.Image.open(image)
            imag = transforms.Resize(size)(img)
            tensor = transformImage(image=imag)
            p = learn.model(tensor)
            predictions[name].append(p.cpu().detach())
            gc.collect()
            torch.cuda.empty_cache()
        del learn
        gc.collect()
        torch.cuda.empty_cache()

    print('Annotating images')
    for image in tqdm(images):
        name = image.split(os.sep)[-1]
        prob, indices = averageVotingEnsemble(predictions[name])
        newMask = getImageFromOut(indices, size, path + os.sep + 'codes.txt')
        img.save(newPath + os.sep + 'Images' + os.sep + 'train' + os.sep + 'new_' + name)
        newMask.save(newPath + os.sep + 'Labels' + os.sep + 'train' + os.sep + 'new_' + name)
    del predictions
    gc.collect()
    torch.cuda.empty_cache()


def omniModelData(path, models, backbones, transformations, size=(320,320)):
    images = sorted(glob.glob(path+os.sep+'unlabeled_images' + os.sep + "*"))
    newPath = path + "_tmp"
    if not os.path.exists(newPath):
        shutil.copytree(path, newPath)
    else:
        raise Exception("The path " + newPath + " already exists")

    predictions = {}
    nClasses = 2
    for model, backbone in zip(models,backbones):
        dls = get_dls(path, size, bs=1)
        learn = getLearner(model, backbone, nClasses, path, dls)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learn.load(model + '_' + backbone,device=device,with_opt=False)
        learn.model.to(device)
        del dls
        gc.collect()
        torch.cuda.empty_cache()
        print('Processing images with ' + model + ' model')
        for image in tqdm(images):
            name = image.split(os.sep)[-1]
            if name not in predictions:
                predictions[name]=[]
            img = PIL.Image.open(image)
            imag = transforms.Resize(size)(img)
            tensor = transformImage(image=imag)
            pn = learn.model(tensor)
            predictions[name].append(pn.cpu().detach())

            im = cv2.imread(image, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            for transform in transformations:
                img = getTransform(transform, im)
                img = PIL.Image.fromarray(img)
                imag = transforms.Resize(size)(img)
                tensor = transformImage(image=imag)
                p = learn.model(tensor)
                p = np.asarray(p.cpu().detach())
                p = p[0]
                axis0 = p.shape[0]
                for i in range(axis0):
                    if i == 0:
                        res = np.expand_dims(getTransformReverse(transform, p[i]), axis=0)
                    else:
                        res = np.append(res, np.expand_dims(getTransformReverse(transform, p[i]), axis=0), axis=0)
                p = np.expand_dims(res, axis=0)
                predictions[name].append(torch.from_numpy(p).cpu().detach())
            gc.collect()
            torch.cuda.empty_cache()
        del learn
        gc.collect()
        torch.cuda.empty_cache()

    print('Annotating images')
    for image in tqdm(images):
        name = image.split(os.sep)[-1]
        prob, indices = averageVotingEnsemble(predictions[name])
        newMask = getImageFromOut(indices, size, path + os.sep + 'codes.txt')
        img.save(newPath + os.sep + 'Images' + os.sep + 'train' + os.sep + 'new_' + name)
        newMask.save(newPath + os.sep + 'Labels' + os.sep + 'train' + os.sep + 'new_' + name)
    del predictions
    gc.collect()
    torch.cuda.empty_cache()
