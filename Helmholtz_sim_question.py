import numpy as np
import matplotlib.pyplot as plt

#Define functions

#Paramaterize the spiral. 
  
def spiral(radius, wire_diameter, turns_per_layer, layers, dt, z_offset):
    '''
    Function that paramaterizes a spiral.
    Inputs:
        radius: radius of the spiral
    
        wire_diameter: Diameter of the wire to be used.
    
        turns_per_layer: number of turns in one layer of the spiral
    
        layers: number of layers in the total coil. Input the actual
            number of layers, not the number of layers minus 1.
    
        dt: number of points to paramaterize the spiral. More points will
            make the spiral smoother and the total approximation more accurate.
            But also more computationally expensive.
    
        z_offset: offset of the spiral in the z direction.
    
    Outputs:
        array of x, y, z values of the spiral. 
        Dimensions: (dt*layers*turns_per_layer, 3)
    '''
    for i in range(layers):
        t = np.linspace(0, 2*np.pi*turns_per_layer, dt) # from 0 to 2pi*turns_per_layer with dt points
        x = (radius+(wire_diameter*i))*np.cos(t)
        y = (radius+(wire_diameter*i))*np.sin(t)
        z = z_offset+t*(wire_diameter/(2*np.pi))
        if i == 0:
            x_total = x
            y_total = y
            z_total = z
        else: # add the new layer to the total array
            x_total = np.concatenate((x_total, x))
            y_total = np.concatenate((y_total, y))
            z_total = np.concatenate((z_total, z))
    return np.column_stack((x_total, y_total, z_total))

#Paramaterize dl of the spiral.
def dl(radius, wire_diameter, turns_per_layer, layers, dt):
    '''
    Function that paramaterizes dl of the spiral.
    Inputs:
        radius: radius of the spiral
    
        wire_diameter: Diameter of the wire to be used.

        turns_per_layer: number of turns in one layer of the spiral
        
        layers: number of layers in the total coil. Input the actual
            number of layers.
        
        dt: number of points to paramaterize the spiral

    Outputs:
        array of dl values of the spiral. Array corresponds to spiral array 
        function. spiral[0] is the position vector while I*dl[0] is equal to 
        the velocity of one small charge element(dq) in the spiral at that 
        position. 
        
        Dimensions: (dt*layers*turns_per_layer, 3)
    '''
    for i in range(layers):
        t = np.linspace(0, 2*np.pi*turns_per_layer, dt) 
        x = -(radius+(layers*wire_diameter))*np.sin(t)
        y = (radius+(layers*wire_diameter))*np.cos(t)
        #array filled with wire_diameter/(2*pi) to match the shape of x and y
        z = np.full(dt, wire_diameter/(2*np.pi))
        if i == 0:
            x_total = x
            y_total = y
            z_total = z
        else:
            x_total = np.concatenate((x_total, x))
            y_total = np.concatenate((y_total, y))
            z_total = np.concatenate((z_total, z))
    return np.column_stack((x_total, y_total, z_total))

def displacement_vector(measured_position, charge_position):
    '''
    Function that calculates the displacement vector from a position to all 
    points on the spiral.
    Inputs:
        spiral: array of x, y, z values of the spiral. 
        Dimensions: (n, 3)
        
        position: position of the point being measured. Can be single point or
        array of points.
        Dimensions: (n, 3)
    
    Outputs: displacement vector from the spiral to all points in the grid.
    Dimensions: 
    '''
    #check if the input is of the form (n, 3)
    if measured_position.shape[1] != 3:
        raise ValueError('Input array measured_position must have three columns')
    if charge_position.shape[1] != 3:
        raise ValueError('Input array charge_position must have three columns')
    for i in range(len(measured_position)):
        x = measured_position[i,0] - charge_position[:,0] #
        y = measured_position[i,1] - charge_position[:,1]
        z = measured_position[i,2] - charge_position[:,2]
        if i == 0:
            x_total = x
            y_total = y
            z_total = z
        else:
            x_total = np.concatenate((x_total, x))
            y_total = np.concatenate((y_total, y))
            z_total = np.concatenate((z_total, z))
    return np.column_stack((x_total, y_total, z_total))

def vector_magnitude_3d(vector):
    '''
    Function that calculates the magnitude of any given three dimensional vector.
    Inputs:
        vector: array of x, y, z values of the vector.
        Dimensions: (n, 3)
        
    
    Outputs: magnitude of the vector or set of vectors.
    Dimensions: (n, 1) This will correspond to the displacement vector array.
    index i of output is equivalent to the magnitude of three dimensional vector 
    at index i of the input array.
    '''
    #check if the input is of the form (n, 3)
    if vector.shape[1] != 3:
        raise ValueError('Input array must have three columns')
    return np.sqrt(vector[:, 0]**2 + 
        vector[:, 1]**2 + vector[:, 2]**2) # shape 

def cross_product(vector1, vector2):
    '''
    Function that calculates the cross product of two vectors.
    Inputs:
        vector1: array of x, y, z values of the first vector.
        Dimensions: (n, 3)
        
        vector2: array of x, y, z values of the second vector.
        Dimensions: (n, 3)
    
    Outputs: cross product of the two vectors.
    Dimensions: (n, 3) This is the cross product of all the possible combinations
    of the two input. Index 0 through length of vector1 - 1 is the cross product
    of all the points of the first vector with the first point of the second vector.

    RUNS SLOW AS SHIT
    '''
    #check if the input is of the form (n, 3)
    if vector1.shape[1] != 3:
        raise ValueError('Input array vector1 must have three columns')
    if vector2.shape[1] != 3:
        raise ValueError('Input array vector2 must have three columns')
    cross = np.zeros((len(vector1), 3))
    for i in range(len(vector1)):
        cross[i, 0] = vector1[i, 1]*vector2[i, 2] - vector1[i, 2]*vector2[i, 1]
        cross[i, 1] = vector1[i, 2]*vector2[i, 0] - vector1[i, 0]*vector2[i, 2]
        cross[i, 2] = vector1[i, 0]*vector2[i, 1] - vector1[i, 1]*vector2[i, 0]
    return cross



def Helm(radius, wire_diameter, turns_per_layer, layers, dt, z_offset,
        grid_spacing, measured_position, I):
    '''Desription: Function that calculates the magnetic field produced by two coils
    in a helmholtz configuration. The magnetic field is calculated at an array of points
    in space with the Biot-Savart law. Is possible to that it will calculate field given
    any line of current and its dl vector. Further testing is needed to confirm this.
    Inputs:
        radius: radius of the spiral

        wire_diameter: Diameter of the wire to be used.

        turns_per_layer: number of turns in one layer of the spiral

        layers: number of layers in the total coil. Input the actual
            number of layers, not the number of layers minus 1.

        dt: number of points to paramaterize the spiral. More points will
            make the spiral smoother and the total approximation more accurate.
            But also more computationally expensive.

        z_offset: offset of the spiral in the z direction.

        grid_spacing: spacing of the grid in the x, y, and z directions. 
        The grid will be a cube with the origin at the center. The grid will
        be from -grid_spacing to grid_spacing in all directions.

        measured_position: array of x, y, z values of the points in space where the
        magnetic field is to be calculated.

        I: current in the wire (A)

    Returns:
        B_vec: array of x, y, z values of the magnetic field at the points in space.
        Indexing corresponds to the index of the measured_position array.
        Both will need to be reshaped to the shape of the grid to be plotted. 
        Dimensions: (m, 3)
    '''
    constants = I*mu_0/(4*np.pi) 
  

    #spiral1: array of x, y, z values of the first spiral.
    #Dimensions: (n, 3) VERY IMPOTANT that all n values are the same for all inputs.
    spiral1 = spiral(radius, wire_diameter, turns_per_layer, layers, dt, z_offset)
    
    #spiral2: array of x, y, z values of the second spiral.
    #Dimensions: (n, 3)
    spiral2 = spiral(radius, wire_diameter, -turns_per_layer, layers, dt, -z_offset)

    #delta factor to be used as the dx in the trapz function
    #len(spiral1) is used as the number of points in the spiral
    delta = 2*np.pi*turns_per_layer*layers/len(spiral1)

    #dl1: array of x, y, z values of the dl vector of the first spiral. Derivative
    #of the position vector of spiral1.
    #Dimensions: (n, 3)
    dl1 = dl(radius, wire_diameter, turns_per_layer, layers, dt)

    #dl2: array of x, y, z values of the dl vector of the second spiral. Derivative
    #of the position vector of spiral2.
    #Dimensions: (n, 3)
    dl2 = dl(radius, wire_diameter, -turns_per_layer, layers, dt)

    #measured_position: array of x, y, z values of the points in space where the
    #magnetic field is to be calculated.
    #Dimensions: (m, 3)
    measured_position = np.linspace(-grid_spacing, grid_spacing, points)

    #initialize the array to store the magnetic field at each point.
    # this index corresponds to the index of the measured_position array.
    # Stores vector components of the magnetic field at each point.
    B_vec = np.zeros((points**2, 3))
    B_mag = np.zeros(points**2)
    B_z_degrees = np.zeros(points**2)

    #for loop that goes through all the points in the measured_position array
    # and calculates the magnetic field at that point
    for i in range(len(measured_position)):
        #takes the first point in the measured_position array. .reshape(1,-1) is used
        #to make the array 2d. (1,-1) means one row and as many columns as needed.
        #this satisfies the input requirements of the displacement_vector function.
        displacement1 = displacement_vector(measured_position[i].reshape(1,-1), spiral1)
        displacement2 = displacement_vector(measured_position[i].reshape(1,-1), spiral2)
        # calculate the magnitude cubed of the displacement vectors.
        # it is uneccessary to store the magnitude as it is a intermediate step.
        # this index corresponds to the index of the displacement vector array,
        # and the position vector array for the spiral. It should be 1d though as it is
        # a scalar value. When dividing by remember to do to each element of vector.

        mag_cubed1 = (vector_magnitude_3d(displacement1))**3
        mag_cubed2 = (vector_magnitude_3d(displacement2))**3
        # transform mag_cubed to a 2d (n,3) array to match the shape of the displacement vector
        # array. This is done to make the division of the displacement vector by the magnitude
        # cubed easier.
        mag_cubed1 = np.column_stack((mag_cubed1, mag_cubed1, mag_cubed1))
        mag_cubed2 = np.column_stack((mag_cubed2, mag_cubed2, mag_cubed2))
        


        # calculate the cross product of dl and the displacement vector
        # this index corresponds to the index of the displacement vector array,
        # and the position vector array for the spiral. It is the cross product for each
        # point in the spiral with the point currently being measured.
        
        cross1 = np.cross(dl1, displacement1)
        cross2 = np.cross(dl2, displacement2)
        # calculate the magnetic field at the point being measured.
        B_vec[i] = constants*np.trapz(cross1/mag_cubed1, dx = delta, axis = 0)
        B_vec[i] += constants*np.trapz(cross2/mag_cubed2, dx = delta, axis = 0)


    return B_vec


#main body of code

#Define constants
I = 500 #current in the wire (A)
mu_0 = 4*np.pi*10**-7 #permeability of free space (T*m/A)

#Define the dimensions of the space and spiral

radius = .3 #radius of the spiral (m)
wire_diameter = .003 #diameter of the wire (m)
turns = 33 #number of turns in the spiral
length = radius+.2 # dimensions of the space to be analyzed (m)
points = 25 #number of points along each of the x, y, and z axes 
dt = 2500 #number of points to paramaterize the spiral
stacks = 10 #number of stacked spirals
z_offset = radius/2 #offset of the spiral in the z direction

#Paramaterize the spiral
#spiral1 = spiral(radius, wire_diameter, turns, stacks, dt, z_offset)
#spiral2 = spiral(radius, wire_diameter, -turns, stacks, dt, -z_offset)

#debug code
#print(len(spiral1))

#Paramaterize dl of the spiral
#dl1 = dl(radius, wire_diameter, turns, stacks, dt)
#dl2 = dl(radius, wire_diameter, -turns, stacks, dt)
#Paramaterize the position of the point being measured
x1 = np.linspace(-length, length, points)
y1 = np.linspace(-length, length, points)
z1 = 0
X1, Y1 = np.meshgrid(x1, y1)

# This is an array of all the points in the grid. x, y, z values of the points
# Dimensions: (points**2, 3)
# index moves through all possible x point of lowest y values then up one step in y
# and moves through all possible x points of that y value.

measured_position_xy = np.column_stack((X1.flatten(), Y1.flatten(), np.full(points**2, z1)))


#Gridpoints to be used for xz plane plot
x2 = np.linspace(-length, length, points)
y2 = 0
z2 = np.linspace(-length, length, points)
X2, Z2 = np.meshgrid(x2, z2)
measured_position_xz = np.column_stack((X2.flatten(), np.full(points**2, y2), Z2.flatten()))

# XY plane calculations
B_vec_xy = Helm(radius, wire_diameter, turns, stacks, dt, z_offset, points, measured_position_xy, I)
B_mag_xy = vector_magnitude_3d(B_vec_xy)
B_z_degrees_xy = np.arccos(B_vec_xy[:,2]/B_mag_xy)*(180/np.pi)

# XZ plane calculations
B_vec_xz = Helm(radius, wire_diameter, turns, stacks, dt, z_offset, points, measured_position_xz, I)
B_mag_xz = vector_magnitude_3d(B_vec_xz)
B_z_degrees_xz = np.arccos(B_vec_xz[:,2]/B_mag_xz)*(180/np.pi)


#reshape the B_vec arrays to the shape of the grid
B_vec_xy = B_vec_xy.reshape(points, points, 3)
B_vec_xz = B_vec_xz.reshape(points, points, 3)

#reshape the B_mag arrays to the shape of the grid
B_mag_xy = B_mag_xy.reshape(points, points, 1)
B_mag_xz = B_mag_xz.reshape(points, points, 1)
#reshape the B_z_degrees array to the shape of the grid
B_z_degrees_xy = B_z_degrees_xy.reshape(points, points, 1)
B_z_degrees_xz = B_z_degrees_xz.reshape(points, points, 1)
print(B_mag_xy.shape)

# Plot the magnetic field topology
circle1 = plt.Circle((0, 0), radius, color='r', fill=False)
circle2 = plt.Circle((0, 0), radius, color='r', fill=False)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Increased height for better visibility
levels = np.linspace(0, 0.05, 10)

# Top row - XY plane plots
axes[0, 0].contourf(X1, Y1, B_mag_xy[:, :, 0], cmap='viridis', levels=levels)
axes[0, 0].set_title('Magnetic Field Magnitude XY Plane')
axes[0, 0].set_xlabel('x (m)')
axes[0, 0].set_ylabel('y (m)')
axes[0, 0].set_xlim(-length, length)
axes[0, 0].set_ylim(-length, length)
axes[0, 0].set_aspect('equal')
axes[0, 0].add_artist(circle1)

axes[0, 1].contourf(X1, Y1, B_z_degrees_xy[:, :, 0], cmap='viridis')
axes[0, 1].set_title('Magnetic Field XY Plane Degrees From Z Axis')
axes[0, 1].set_xlabel('x (m)')
axes[0, 1].set_ylabel('y (m)')
axes[0, 1].set_xlim(-length, length)
axes[0, 1].set_ylim(-length, length)
axes[0, 1].set_aspect('equal')
axes[0, 1].add_artist(circle2)


axes[0, 2].quiver(X1, Y1, B_vec_xy[:, :, 0], B_vec_xy[:, :, 1],
                color='blue',
                scale=1)
axes[0, 2].grid(True)
axes[0, 2].set_title('Magnetic Vector Field XY Plane')
axes[0, 2].set_xlabel('x (m)')
axes[0, 2].set_ylabel('y (m)')
axes[0, 2].set_xlim(-length, length)
axes[0, 2].set_ylim(-length, length)
axes[0, 2].set_aspect('equal', adjustable='box')
axes[0, 2].add_artist(plt.Circle((0, 0), radius, color='r', fill=False))

# Bottom row - XZ plane plots

axes[1, 0].contourf(X2, Z2, B_mag_xz[:, :, 0], levels=levels, cmap='viridis')
axes[1, 0].set_title('Magnetic Field Magnitude XZ Plane')
axes[1, 0].set_xlabel('x (m)')
axes[1, 0].set_ylabel('z (m)')  # Changed from 'y (m)' to 'z (m)'
axes[1, 0].set_xlim(-length, length)
axes[1, 0].set_ylim(-length, length)
axes[1, 0].set_aspect('equal')

axes[1, 1].contourf(X2, Z2, B_z_degrees_xz[:, :, 0], cmap='viridis')
axes[1, 1].set_title('Magnetic Field XZ Plane Degrees From Z Axis')
axes[1, 1].set_xlabel('x (m)')
axes[1, 1].set_ylabel('z (m)')  # Changed from 'y (m)' to 'z (m)'
axes[1, 1].set_xlim(-length, length)
axes[1, 1].set_ylim(-length, length)
axes[1, 1].set_aspect('equal')


# Only plot once with the full grid and filtered vectors
axes[1, 2].quiver(X2, Z2, B_vec_xz[:, :, 0], B_vec_xz[:, :, 2],
                color='blue',
                scale=1)
axes[1, 2].grid(True)
axes[1, 2].set_title('Magnetic Vector Field XZ Plane')
axes[1, 2].set_xlabel('x (m)')
axes[1, 2].set_ylabel('z (m)')
axes[1, 2].set_xlim(-length, length)
axes[1, 2].set_ylim(-length, length)
axes[1, 2].set_aspect('equal', adjustable='box')  # Force the box to adjust

# Add colorbars
plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Field Strength (T)')
plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Angle (degrees)')
plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Field Strength (T)')
plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Angle (degrees)')

plt.tight_layout()
plt.show()


# print the magnitude of the magnetic field at the origin
print(B_mag_xy[int(points/2), int(points/2)])
print(B_mag_xz[int(points/2), int(points/2)])
print(B_vec_xy[int(points/2), int(points/2)])
print(B_vec_xz[int(points/2), int(points/2)])
#     


'''r_mag = vector_magnitude_3d(r)
r_mag_cubed = r_mag**3
cross = np.cross(dl, r)
print(cross.shape)
print(r_mag.shape'''
