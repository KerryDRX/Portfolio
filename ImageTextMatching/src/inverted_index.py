# ====================================================================
# This script takes the image tag information given by the file
#   ../dataset/image_info.json
# and transforms it into a dictionary with the inverted index,
# and stores the dictionary into
#   ../dataset/tag2picid.json
# for further usage.
#
# The image_info.json contains a list of dictionaries, each of
# which represents an image. The keys include the picture id and
# the tag term of that image.
#
# The tag2picid.json stores dictionary, each item represents a
# tag term. A key in the dictionary is the tag term itself, and
# the value is a list of picture ids of those images with this
# tag term.
# ====================================================================
import json
# ====================================================================
# This function constructs and saves a dictionary
# which maps an image tag term to the list of
# image indices with that tag term.
# ====================================================================
def inverted_index(path_input, path_output):
    # A dictionary that maps a tag term to
    # its corresponding image indices
    tag2picid = {}
    with open(path_input, encoding='utf8') as f:
        for line in f:
            image_info = json.loads(line.strip('\n'))
            tag = image_info['tags_term']
            pic_id = image_info['pic_id']
            if tag not in tag2picid:
                tag2picid[tag] = []
            tag2picid[tag].append(pic_id)
    with open(path_output, 'w', encoding='utf8') as f:
        json.dump(tag2picid, f)


if __name__ == '__main__':
    inverted_index('../dataset/image_info.json', '../dataset/tag2picid.json')
