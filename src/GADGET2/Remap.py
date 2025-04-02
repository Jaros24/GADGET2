import shutil

import h5py
import numpy as np
import tqdm


def remap_pad_numbers(input_file:str,
                      output_file:str,
                      flatlookup_file:str):
    '''
    Copy input file to output file, and then renumber the pads based on flatlookup_file
    '''
    flat_lookup = np.loadtxt(flatlookup_file, delimiter=',', dtype=int)
    chnls_to_pad = {} #maps tuples of (asad, aget, channel) to pad number
    for line in flat_lookup:
        chnls = tuple(line[0:4])
        pad = line[4]
        chnls_to_pad[chnls] = pad
    
    shutil.copyfile(input_file, output_file)
    with h5py.File(output_file, 'r+') as file:
        first_event_num, last_event_num = int(file['meta']['meta'][0]), int(file['meta']['meta'][2])
        for event_num in tqdm.tqdm(range(first_event_num, last_event_num+1)):
            data = file['get']['evt%d_data'%event_num]
            for line_num in range(len(data)):
                chnl_info = tuple(data[line_num, 0:4])
                if chnl_info in chnls_to_pad:
                    data[line_num, 4] = chnls_to_pad[chnl_info]
                else:
                    print('Event #' + str(event_num) + ' channel information ' + str(chnl_info) + ' not in pad mapping!')


def remap_cobo_and_asads(input_file:str,
                         output_file:str,
                         remap_map:dict={(0,0):(0,0),(0,1):(0,1),(1,0):(0,2),(1,1):(0,3)}):
    '''
    This function can be used to make a new pad lookup table from an old one.

    remap_map defines this mapping by (old_cobo, old_asad)->(new_cobo, new_asad)
    input_file is the channel mapping to remap, and output_file is the new map to create
    '''
    with open(input_file) as input_file, open(output_file, 'w') as output_file:
        for line in input_file:
            if len(line) == 0:
                continue
            cobo, asad, aget, chnl, pad = np.fromstring(line, sep=',')
            cobo, asad = remap_map[cobo, asad]
            output_file.write('%d, %d, %d, %d, %d\n' % (cobo, asad, aget, chnl, pad))