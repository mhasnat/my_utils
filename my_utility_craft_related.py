import numpy as np
# import craft functions
from craft_text_detector import read_image, load_craftnet_model, load_refinenet_model, get_prediction
from craft_text_detector import export_detected_regions, export_extra_results, empty_cuda_cache

# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)

def get_craft_detected_region(image, pad_pixels=0):
    # Apply Prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )
    
    # Get bounding box
    (x1, y1, x2, y2) = get_text_region_craft(prediction_result, image.shape, pad_pixels=pad_pixels)
    
    return (x1, y1, x2, y2)

def get_text_region_craft(prediction_result, image_shape, pad_pixels=None):
    all_rects = []
    for t_poly in prediction_result["polys"]:
        #print(t_poly)
        all_rects.append([t_poly[0][0], t_poly[0][1], t_poly[2][0], t_poly[2][1]])

    all_rects = np.asarray(all_rects, dtype=np.int)    
    x1 = np.min(all_rects[:,0])
    y1 = np.min(all_rects[:,1])
    x2 = np.max(all_rects[:,2])
    y2 = np.max(all_rects[:,3])
    
    if pad_pixels is not None:
        num_pix_x = int(image_shape[1]*(pad_pixels/image_shape[1]))
        num_pix_y = int(image_shape[0]*(pad_pixels/image_shape[0]))
        
        x1 = max(0, x1-num_pix_x)
        y1 = max(0, y1-num_pix_y)
        x2 = min(image_shape[1], x2+num_pix_x)
        y2 = min(image_shape[0], y2+num_pix_y)

    return (x1, y1, x2, y2)