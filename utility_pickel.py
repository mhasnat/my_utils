import pickle

def save_data_with_pickel(file_name, info_list):
    with open(file_name, 'wb') as f:
        pickle.dump(info_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Info saved to :' + file_name)
        
def load_data_from_pickel(file_name):
    with open(file_name, 'rb') as f:
        all_info = pickle.load(f)
    return all_info