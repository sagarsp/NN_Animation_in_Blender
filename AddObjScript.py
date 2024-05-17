import bpy
import math
import numpy as np
from mathutils import Vector

#bpy.context.space_data.params.filename = "nn_anim.blend"
#bpy.context.space_data.context = 'WORLD'
#bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value[0] = 0
#bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value[1] = 0
#bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value[2] = 0


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

images = 50
frames_per_images = 20
img_dim_h = 28
img_dim_v = 28
nn_l1 = 10
nn_l2 = 10

frame_count = images*frames_per_images
bpy.context.scene.frame_end = frame_count

X_imgs = np.load("/home/sagar/Blender_work/mnist_nn_train/dump_data/inputs.npy")
A2_50 = np.load("/home/sagar/Blender_work/mnist_nn_train/dump_data/outputs.npy")
W1_mult_X = np.load("/home/sagar/Blender_work/mnist_nn_train/dump_data/weights1.npy")
W2_mult_A1 = np.load("/home/sagar/Blender_work/mnist_nn_train/dump_data/weights2.npy")
A1_50 = np.load("/home/sagar/Blender_work/mnist_nn_train/dump_data/activations1.npy")

#X_imgs = X_imgs * 1.2
W1_mult_X = W1_mult_X * 1.2
W2_mult_A1 = W2_mult_A1 * 1.2
X_imgs = np.minimum(X_imgs, 1)
W1_mult_X = np.minimum(W1_mult_X, 1)
W2_mult_A1 = np.minimum(W2_mult_A1, 1)

print(np.shape(X_imgs))
print(np.shape(A2_50))
print(np.shape(W1_mult_X))
print(np.shape(W2_mult_A1))
print(np.shape(A1_50))

X_imgs_T = X_imgs.T
A1_50_T = A1_50.T
A2_50_T = A2_50.T


## Copied from https://github.com/DanieliusKr/neural-network-blender/blob/main/blender_script.py
def set_material_colors(obj, values, frame_num):
    # Create a new material for the object
    mat = bpy.data.materials.new(name="ColorMat")
    mat.use_nodes = True
    emmNode = mat.node_tree.nodes.new(type="ShaderNodeEmission")
    in_1 = mat.node_tree.nodes["Material Output"].inputs["Surface"]
    out_1 = emmNode.outputs[0]
    #out_1 = mat.node_tree.nodes["Emission"].outputs[0]

    mat.node_tree.links.new(in_1,out_1)
    #bpy.data.objects[“Cube”].active_material = bpy.data.materials[matName]
    ###
    obj.data.materials.append(mat)
    
    # Loop through the values and set the material color for each one
    for i, val in enumerate(values):
        # Create a new color and set it on the material
        color = (val, val, val, 1.0) 
        #mat.diffuse_color = color
        emmNode.inputs[0].default_value = (val,val,val,val)
        #mat.node_tree.nodes["Emission"].inputs[0].default_value = (val,val,val,val)
        #mat.node_tree.nodes["Emission"].inputs[1].default_value = 5*val
        emmNode.inputs[1].default_value = 3*val
        # Insert a keyframe for the material color every frame_num frames
        frame = i * frame_num
        #mat.keyframe_insert(data_path="diffuse_color", frame=frame)
        emmNode.inputs[0].keyframe_insert(data_path="default_value", frame=frame)
        emmNode.inputs[1].keyframe_insert(data_path="default_value", frame=frame)
        #mat.node_tree.nodes["Emission"].inputs[0].keyframe_insert(data_path="default_value", frame=frame)
        #mat.node_tree.nodes["Emission"].inputs[1].keyframe_insert(data_path="default_value", frame=frame)

def create_plane_grid(n, colours, spacing, size, x_position, delay):
    plane_locations = []
    for i in range(n[0]):
        for j in range(n[1]):
            x_i = x_position
            y_i = img_dim_h - (j*2)
            z_i = img_dim_v - (i*2)

            # Create a new plane
            bpy.ops.mesh.primitive_plane_add(size=1.5,location=(x_i,y_i,z_i),rotation=(0,3.14/2,0))
            plane = bpy.context.active_object
            
            plane_locations.append(Vector((x_i, y_i, z_i, 0)))
            c_index = j + (img_dim_h*i)
            #print(i,j,c_index)
            set_material_colors(plane, colours[:,c_index], frames_per_images+delay)
    return plane_locations

def create_neurons_grid(n, colours, radius_of_spread, radius_of_sphere, x_position,delay):
    neurons_locations = []
    neurons_count = n[0]
    x_i2 = x_position
    for i2 in range(0,neurons_count):
        theta = i2 * 2 * 3.14 / neurons_count
        y_i2 = radius_of_spread * math.cos(theta)
        z_i2 = radius_of_spread * math.sin(theta)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius_of_sphere, location=(x_i2,y_i2,z_i2))
        n_sphere = bpy.context.active_object
        neurons_locations.append(Vector((x_i2,y_i2,z_i2, 0)))
        set_material_colors(n_sphere, colours[:,i2], frames_per_images+delay)
        
    return neurons_locations

def create_cube_grid(n, colours, spacing, size, x_position, delay):
    cube_locations = []
    neurons_count = n[0]
    x_i4 = x_position
    z_i4 = 0
    # Add cube in python
    for i in range(0,n[0]):
        y_i4 = -(n[0]*spacing/2) + (i*spacing)
        bpy.ops.mesh.primitive_cube_add(size=3,location=(x_i4,y_i4,z_i4))
        n_cube = bpy.context.active_object
        cube_locations.append(Vector((x_i4,y_i4,z_i4, 0)))
        set_material_colors(n_cube, colours[:,i], frames_per_images+delay)
        
    return cube_locations


def create_curve_object(point_a, point_b, colours, thickness,delay):
    # Create a new curve
    curve_data = bpy.data.curves.new('Curve', 'CURVE')
    curve_data.dimensions = '3D'
    
    # Create a new spline
    spline = curve_data.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = point_a
    spline.points[1].co = point_b
    
    # Set the thickness of the curve
    curve_data.bevel_depth = thickness
    
    # Create a new object and link it to the scene
    obj = bpy.data.objects.new('Curve', curve_data)
    bpy.context.scene.collection.objects.link(obj)
    
    # Set the object to use the curve as its data
    obj.data = curve_data
    #set_material_colors(obj, colours, 20)
    set_material_colors(obj, colours, frames_per_images+delay)

    return obj

def get_thickness (v):
    thickness_min = 0.03
    thickness_max = 0.08
    if v < 0:
        return thickness_min
    elif v > 1:
        return thickness_max
    else:
        return round((0.05 * v) + thickness_min,3)

def create_curves_between_points(point_list_1, point_list_2, colours, thickness,delay):
    # Loop through all combinations of points in the two lists
    for i, point_a in enumerate(point_list_1):
        for j, point_b in enumerate(point_list_2):
            #print(i,j)
            # Create a curve object between each pair of points
            colours_for_j_i = colours[:,j,i]
            avg_col_val_over_given_samples = np.average(colours_for_j_i)
            thickness = get_thickness(avg_col_val_over_given_samples)
            create_curve_object(point_a, point_b, colours[:,j,i], thickness,delay)
            


x_i1=0
x_i2 = 50
x_i3 = 100
#x_i4 = 100
radius_of_spread = 21
radius_of_neuron_sphere = 1
neurons_level_2 = 10
plane_locations = create_plane_grid((img_dim_v,img_dim_h), X_imgs_T, 2, 1.5, x_i1, 0)
layer1_neurons_locations = create_neurons_grid([nn_l1], A1_50_T, radius_of_spread, radius_of_neuron_sphere, x_i2,0)
#layer2_neurons_locations = create_neurons_grid([nn_l2], A2_50_T, radius_of_spread, radius_of_neuron_sphere, x_i3,0)
cube_locations = create_cube_grid([nn_l2], A2_50_T, 4, 3, x_i3,0)

#SAGAR# create_curves_between_points(plane_locations, layer1_neurons_locations, W1_mult_X, 0.05,0)
#SAGAR# create_curves_between_points(layer1_neurons_locations, cube_locations, W1_mult_X, 0.05,0)


#Add Empty
bpy.ops.object.empty_add()
empty = bpy.context.active_object

#Animate Empty
empty.location.x = 60
empty.keyframe_insert("location",frame=1)
empty.location.x = 50
empty.keyframe_insert("location",frame=frame_count/2)
empty.location.x = 60
empty.keyframe_insert("location",frame=frame_count)

# Add circle
bpy.ops.curve.primitive_bezier_circle_add(radius=200,location=(20,0,0),rotation=(-3.14,0,3.14/2))
circle = bpy.context.active_object


#Add Camera
bpy.ops.object.camera_add()
camera = bpy.context.active_object

#Animate Camera
camera.location.x=0
camera.location.y=0
camera.location.z=0

# Follow Path Constraint
bpy.ops.object.constraint_add(type='FOLLOW_PATH')
camera.constraints["Follow Path"].target = circle

# Animate Button
bpy.ops.constraint.followpath_path_animate({
    'constraint':camera.constraints["Follow Path"]
    }, constraint='Follow Path',length=frame_count)

bpy.ops.object.constraint_add(type='TRACK_TO')
camera.constraints["Track To"].target = empty

bpy.context.scene.camera = camera
bpy.context.scene.eevee.use_bloom = True
#bpy.context.space_data.shading.type = 'RENDERED'

print("done")