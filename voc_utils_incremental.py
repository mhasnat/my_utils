import torch
from vision import torchvision

import transforms as T

class ConvertVOCtoCOCO(object):
    def __init__(self, CLASSES):
        self.CLASSES = CLASSES
        
    def __call__(self, image, target):
        # return image, target
        anno = target['annotations']
        filename = anno["filename"].split('.')[0]
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        ishard = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
            
        for obj in objects:
            if obj['name'] in self.CLASSES:
                classes.append(self.CLASSES.index(obj['name']))
            else:
                continue
                
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            
            ishard.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        ishard = torch.as_tensor(ishard)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8) #convert filename in int8
        
        return image, target

class VOCDetection(torchvision.datasets.VOCDetectionCustom):
    def __init__(self, img_folder, year, image_set, transforms, selected_class_names=None):      
        super(VOCDetection, self).__init__(img_folder,  year, image_set, selected_class_names)
        self._transforms = transforms
        #self.CLASSES = CLASSES

    def __getitem__(self, idx):
        #print('=='*10)        
        #print(idx)
        img, target, img_name = super(VOCDetection, self).__getitem__(idx)        
        #print(target)
        
        target = dict(image_id=idx, annotations=target['annotation'])

        #print(target)        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # img = img[[2, 1, 0],:]

        return img, target, img_name

def get_voc(root, image_set, transforms, CLASSES, incremental_classes=None, use_old_data=False):
    if incremental_classes is None:
        t = [ConvertVOCtoCOCO(CLASSES)]
        incremental_classes = CLASSES
    else:
        t = [ConvertVOCtoCOCO(CLASSES+incremental_classes)]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)
    
    if use_old_data:
        selected_class_names = CLASSES+incremental_classes
    else:
        selected_class_names = incremental_classes
        
    dataset = VOCDetection(img_folder=root, year='2007', 
                           image_set=image_set, transforms=transforms, 
                           selected_class_names=selected_class_names)
    
    return dataset