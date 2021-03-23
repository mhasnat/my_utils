def write_yolo_results(det_, path_res_save, t_im_file, CLASSES_NAMES, 
                       is_xywh=True, start_from_zero=True):
    t_res_file = path_res_save + t_im_file.split('/')[-1].split('.')[0]+'.txt'

    t_fid = open(t_res_file, 'w')
    for t_ in det_:
        t_str = str(CLASSES_NAMES.index(t_[0]))
        t_str += ' ' + str(t_[1]) # confidense
        
        if not is_xywh:
            t_str += ' ' + ' '.join([str(int(ts_+1)) for ts_ in t_[2]])
        else:
            t_str += ' ' + str(int(t_[2][0])) # x
            t_str += ' ' + str(int(t_[2][1])) # y
            t_str += ' ' + str(int(t_[2][2] - t_[2][0])) # w
            t_str += ' ' + str(int(t_[2][3] - t_[2][1])) # h
            
        t_str += '\n'
        print(t_str)
        t_fid.writelines(t_str)    
    t_fid.close()