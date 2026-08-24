import numpy as np
#image of dimension x_total x y_total
x_total = 224
y_total = 224

def get_left_up_corner_in_pixels(bounding_box):
    x_center = bounding_box[0]*x_total
    y_center = bounding_box[1]*y_total
    height = bounding_box[2]*x_total
    width = bounding_box[3]*y_total
    x_left_up_corner = x_center - 0.5*(height)
    y_left_up_corner = y_center - 0.5*(width)
    print("Values in pixels (unnormalized):")
    print("x_left_up_corner=", x_left_up_corner)
    print("y_left_up_corner=", y_left_up_corner)
    print('x_center=', x_center)
    print('y_center=', y_center)
    print('height=',height)
    print('width=',width)
    return x_left_up_corner, y_left_up_corner

#bounding box of first object: x_center, y_center, height, width
print('First object:')
normalized_bb1 = np.array([0.2, 0.7, 0.2, 0.1])
x_left_up_corner, y_left_up_corner = get_left_up_corner_in_pixels(normalized_bb1)

#bounding box of second object: x_center, y_center, height, width
print('Second object:')
normalized_bb2 = np.array([0.8, 0.7, 0.2, 0.1])
x_left_up_corner, y_left_up_corner = get_left_up_corner_in_pixels(normalized_bb2)
