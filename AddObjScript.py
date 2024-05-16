import bpy
import math
import numpy as np

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

images = 50
frames_per_images = 10
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

print(np.shape(X_imgs))
print(np.shape(A2_50))
print(np.shape(W1_mult_X))
print(np.shape(W2_mult_A1))
print(np.shape(A1_50))
# Add cube in python
for i in range(0,img_dim_v):
    for j in range(0,img_dim_h):
        x_i = 0
        y_i = img_dim_h - (j*2)
        z_i = img_dim_v - (i*2)
        
        bpy.ops.mesh.primitive_plane_add(size=1.5,location=(x_i,y_i,z_i),rotation=(0,3.14/2,0))
        #bpy.ops.mesh.primitive_cube_add(size=1.5,location=(x_i,y_i,z_i))
        cb = bpy.context.active_object


#bpy.ops.mesh.primitive_uv_sphere_add(radius=1, enter_editmode=False, align='WORLD', location=(19.8441, -2.59942, -0.597318), scale=(1, 1, 1))
x_i2 = 35
radius_of_spread = 21
neurons_level_2 = 10
for i2 in range(1,neurons_level_2+1):
    theta = (i2-1) * 2 * 3.14 / neurons_level_2
    y_i2 = radius_of_spread * math.cos(theta)
    z_i2 = radius_of_spread * math.sin(theta)
    nn2 = bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(x_i2,y_i2,z_i2))

x_i3 = 65
radius_of_spread_3 = 21
neurons_level_3 = 10
for i3 in range(1,neurons_level_3+1):
    theta = (i3-1) * 2 * 3.14 / neurons_level_3
    y_i3 = radius_of_spread_3 * math.cos(theta)
    z_i3 = radius_of_spread_3 * math.sin(theta)
    nn3 = bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(x_i3,y_i3,z_i3))


x_i4 = 100
z_i4 = 0
# Add cube in python
for i in range(1,10):
    y_i4 = -20 + (i*4)
    cb4 = bpy.ops.mesh.primitive_cube_add(size=3,location=(x_i4,y_i4,z_i4))

#Add Empty
bpy.ops.object.empty_add()
empty = bpy.context.active_object

#Animate Empty
empty.location.x = 60
empty.keyframe_insert("location",frame=1)
empty.location.x = 40
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

#camera.location.x=200
#camera.location.y=0
#camera.location.z=0

