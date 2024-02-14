from pathlib import Path
import torch
import PIL
from torchvision import transforms
import cv2
from semtorch import get_segmentation_learner


def transformImage(image):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image_aux = image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return my_transforms(image_aux).unsqueeze(0).to(device)


def getImageFromOut(output,size,codesFile):
    vocabs={'Background': [0,[0]], 'Foreground': [1,[255]]}
    mask = np.array(output.cpu())
    mask2=mask.copy()

    for vocab in vocabs.keys():
        mask2[mask==vocabs[vocab][0]]=vocabs[vocab][1][0]

    mask2=np.reshape(mask2,size)
    return Image.fromarray(mask2.astype('uint8'))

def getTransformReverse(transform, image):
    if transform=="H Flip":
        return cv2.flip(image,0)
    elif transform=="V Flip":
        return cv2.flip(image,1)
    elif transform=="H+V Flip":
        return cv2.flip(image,-1)
    else: return image


def averageVotingEnsemble(predictions):
    softOutput=torch.nn.Softmax(1)
    output=softOutput(predictions[0])
    for l in predictions:
        output=output+softOutput(l)
    output=output-softOutput(predictions[0])
    output=output/len(predictions)
    finalPredictions = torch.max(output,1)
    return finalPredictions



def getTransform(transform, image):
    if transform=="H Flip":
        return cv2.flip(image,0)
    elif transform=="V Flip":
        return cv2.flip(image,1)
    elif transform=="H+V Flip":
        return cv2.flip(image,-1)
    elif transform=="Blurring":
        return cv2.blur(image,(5,5))
    elif transform=="Gamma":
        invGamma = 1.0
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        return cv2.LUT(image, table)


get_y_fn = lambda x: Path(str(x).replace("Images","Labels"))

def get_dls(path,size=(320, 320),bs=4):
    vocabs={'Background': [0,[0]], 'Foreground': [1,[255]]}
    trainDB = DataBlock(blocks=(ImageBlock, MaskBlock(vocabs.keys())),
                   get_items=partial(get_image_files,folders=['train','valid']),
                   get_y=get_y_fn,
                   splitter=FuncSplitter(ParentSplitter),
                   item_tfms=[Resize(size), TargetMaskConvertTransform(), transformPipeline()],
                   batch_tfms=Normalize.from_stats(*imagenet_stats)
                  )
    dls = trainDB.dataloaders(path+os.sep+'Images', bs=bs)
    return dls

def ParentSplitter(x):
    return Path(x).parent.name=='valid'


class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0

    def __init__(self, aug):
        self.aug = aug

    def encodes(self, x):
        img, mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])

def transformPipeline():
    transforms = Compose([
        HorizontalFlip(p=0.5),
        Rotate(p=0.40, limit=10), GridDistortion()
    ], p=1)

    transforms = SegmentationAlbumentationsTransform(transforms)
    return transforms


class TargetMaskConvertTransform(ItemTransform):
    def __init__(self):
        super()

    def encodes(self, x):
        img, mask = x

        # Convert to array
        mask = np.array(mask)
        mask2=mask.copy()
        mask2[mask == 0] = 0
        mask2[mask == 255] = 1

        # Back to PILMask
        mask = PILMask.create(mask2)
        return img, mask

def train_learner(learn,epochs,freeze_epochs=2):
    learn.fit_one_cycle(freeze_epochs)
    learn.unfreeze()
    learn.lr_find(show_plot=False)
    lr=learn.recorder.lrs[np.argmin(learn.recorder.losses)]
    if lr<1e-05:
        lr=1e-03
    learn.fit_one_cycle(epochs,lr_max=slice(lr/100,lr))


def getLearner(model,backbone,numClasses,path,dls):
    metric = [Dice(), JaccardCoeff(), foreground_acc]
    save = SaveModelCallback(monitor='dice', fname=model+'_'+backbone)
    early = EarlyStoppingCallback(monitor='dice', patience=5)
    if model=='U-Net':
        learn = get_segmentation_learner(dls=dls, number_classes=numClasses, segmentation_type="Semantic Segmentation",
                                         architecture_name="unet", backbone_name=backbone,
                                         metrics=metric, wd=1e-2,cbs=[early, save],
                                         pretrained=True, normalize=True, path=path)
    elif model=='DeepLab':
        learn=get_segmentation_learner(dls=dls, number_classes=numClasses, segmentation_type="Semantic Segmentation",
                                 architecture_name="deeplabv3+", backbone_name=backbone,
                                 metrics=metric,wd=1e-2,cbs=[early, save],
                                 pretrained=True,normalize=True,path=path)
    elif model=='HRNet':
        learn=get_segmentation_learner(dls=dls, number_classes=numClasses, segmentation_type="Semantic Segmentation",
                                 architecture_name="hrnet", backbone_name='hrnet_w48',
                                 metrics=metric,wd=1e-2,cbs=[early, save],
                                 pretrained=True,normalize=True,path=path)
    return learn
