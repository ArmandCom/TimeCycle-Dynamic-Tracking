# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)

# python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json --base_path /data/Ponc/tracking/JPEGImages/480p/nhl/
# --------------------------------------------------------
import glob
import pickle 
from tools.test import *
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from shapely.geometry import Polygon
from shapely.geometry import asPolygon
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/balls', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


def compute_intersection_by_pairs(polygons, sorted_scores, intersection_th = 0.55):
    from shapely.geometry import Polygon
    from shapely.geometry import asPolygon

    results = []
    for i in range(polygons.shape[0]//2):
        pol_a = asPolygon(polygons[i, :, :])
        pol_b = asPolygon(polygons[i + 1, :, :])
        intersection_area = pol_a.intersection(pol_b)
        iou = intersection_area.area/(pol_a.area+pol_b.area)
        if iou >= intersection_th :
            if(sorted_scores[i] >= sorted_scores[i+1]):
                results.append( [polygons[i,:,:] , sorted_scores[i]] )
            else:
                results.append( [polygons[i+1,:,:] , sorted_scores[i]] )
        else:
            if(sorted_scores[i] >= sorted_scores[i+1]):
                results.append( [polygons[i,:,:] , sorted_scores[i]] )
            else:
                results.append( [polygons[i+1,:,:] , sorted_scores[i]] )
    return results


def filter_bboxes(rboxes, k, c=10):
    """
    rboxes: list, contains: [(np.array(polygon), score),(), ... , ()]
    """
    num_bxs = len(rboxes)
    if(c == 0): # Evita bucle infinit
        return rboxes
    if(num_bxs <= 1):
        return rboxes
    if(num_bxs%2 != 0):
        # We compare by pairs so we need an odd number of boxes
        del rboxes[-1]
        num_bxs = len(rboxes)

    polygons = np.zeros((num_bxs, 4, 2)) # (num_bxs, 8 (points of a polygon, 2 (coordinates of each point)))
    scores = np.zeros(num_bxs)

    for i, (box, score) in enumerate(rboxes):
        box_mat = box.reshape(4,2)
        polygons[i, :, :] = box_mat
        scores[i] = score 
    
    # Compute centroids
    centroids = np.mean(polygons, axis=1)
    indexes_x = np.argsort(centroids[:, 0]) # Ascending order
    indexes_y = np.argsort(centroids[:, 1])
    centroids_x = centroids[indexes_x, 0]
    centroids_y = centroids[indexes_y, 1]
    
    xmax, xmin = centroids_x[-1], centroids_x[0]
    ymax, ymin = centroids_y[-1], centroids_y[0]
    intersection_th = 0.9 # If 2 boxes overlap more than 0.4 the one with max score will remain
    
    if( (xmax - xmin) >= (ymax - ymin) ):
        # Make pairs according to x axis (indexes_x)
        polygons = polygons[indexes_x]
        sorted_scores = scores[indexes_x]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores)
    else:
        # Make pairs according to y axis (indexes_y)
        polygons = polygons[indexes_y]
        sorted_scores = scores[indexes_y]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores)
    
    if(len(rboxes) != 1 and len(rboxes) > k):
        c -= 1
        rboxes = filter_bboxes(rboxes, k, c)
    
    return rboxes
            

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    
    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    
    ims = [cv2.imread(imf) for imf in img_files]
    
    try:
        # init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        init_rect = (737,165, 76, 76) # balls noia
        init_rect = (75,388, 200, 79) # Seagull
        # init_rect = (536,105, 41, 40) # juggling-easy
        init_rect = (437, 306, 115,50) # Eagles
        init_rect = (737, 374, 100, 156) # NHL
        init_rect = (1027, 259, 57, 100) # NFL
        init_rect = (422, 235, 49, 95) # warmup_capitals
        # init_rect = (71, 100, 178-71, 314-100) # warmup_edmonton oilers
        # init_rect = (419, 356, 77, 131) # austrian_hockey_warmup
        # init_rect = (1145, 253, 135, 294) # philadelphia_warmup

        x, y, w, h = init_rect

    except:
        exit()

    toc = 0
    trajs = np.zeros((len(ims),2))
    print("num images ",len(ims))
    ssa= True
    all_bboxes = []
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        frame_boxes = []
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            # state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)
            state, bboxes, rboxes = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            
            print("---- FRAME ",f," -------")
            print("num_bxs = ", len(rboxes))
            rboxes = filter_bboxes(rboxes,5, c=10*len(rboxes)) #puto recursiu, funciona be
            print("num_filtered = ", len(rboxes), " - ", rboxes[0][1])

            # dibuixar nomes la bbox filtrada si esta fora de la bona (state)
            locaux_bona = np.int0(state['ploygon'].flatten()).reshape((-1,2))
            locaux_filt = rboxes[0][0]
            pol_a = asPolygon(locaux_bona)
            pol_b = asPolygon(locaux_filt)
            intersection_area = pol_a.intersection(pol_b)
            if((intersection_area.area/(pol_a.area + pol_b.area)) <= 0.2):
                for i in range(len(rboxes)):
                    location = rboxes[i][0].flatten()
                    location = np.int0(location).reshape((-1, 1, 2))
                    cv2.polylines(im, [location], True, (255, 255, 0), 1)
                    traj = np.average(location, axis=0)[0]
                    frame_boxes.append([traj])
                    cv2.circle(im, (int(traj[0]), int(traj[1])) , 3, (255,255,0), 2)
               
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            laloc = np.int0(location).reshape((-1,1,2))
            traj = np.int0(np.average(laloc, axis=0)[0])
            frame_boxes.append([traj])
            for i in range(laloc.shape[0]):
                cv2.circle(im, (laloc[i,0,0], laloc[i,0,1]),6,(0,0,255))
                cv2.circle(im, (traj[0], traj[1]) , 3, (255,255,0), 2)
            
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.putText(im,str(state['score']),(50,50),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0))
            cv2.imwrite('/data/Ponc/tracking/results/capitals_warmup_carlson_debug/'+str(f)+'.jpeg', im)
            all_bboxes.append(frame_boxes)
        toc += cv2.getTickCount() - tic
    
    with open('/data/Ponc/tracking/grr.obj','wb') as fil:
        pickle.dump(all_bboxes, fil)
    
    toc /= cv2.getTickFrequency()
    fps = f / toc
    
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))