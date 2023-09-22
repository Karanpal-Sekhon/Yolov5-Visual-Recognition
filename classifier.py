import numpy as np
import torch
from tqdm import tqdm


def classify_and_detect(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelpath = "best.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path = modelpath, _verbose = False,
    device = device)

    images = images.reshape((N, 64,64,3))
    loader = torch.utils.data.DataLoader(images, batch_size = 1, shuffle = False)

    
    for i, img in tqdm(enumerate(loader)):
        img = img.numpy()[0]
        outputs = model(img).pred[0].cpu().tolist()
        
    

        if len(outputs) != 0:
            if len(outputs) == 1:
                outputs.append(outputs[0])
            outputs = outputs[:2]
            #swapping
            if outputs[0][5] > outputs[1][5]:
                temp = outputs[0]
                outputs[0] = outputs[1]
                outputs[1] = temp


            pred_class[i] = [outputs[0][5], outputs[1][5]]

            bbxs = [0,0]


            for j, box in enumerate(outputs):
                x = round((box[0]+box[2])/ 2)
                y = round((box[1]+box[3])/ 2)
                miny = y-14
                minx = x-14
                maxy = y+14
                maxx = x+14

                bbxs[j] = [miny, minx, maxy, maxx]

            pred_bboxes[i] = bbxs

    return pred_class, pred_bboxes
