### File of helper functions for implementation of FastSLAM2.0 
# @author Adam Oushervitch
# @author Benjamin Asch
# @author Leo Jiang
# @author Timothy Park

# notes: most of these are math that we need to test with unit tests

import numpy as np
import math
from DataStructs import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# TODO: Get better resampling algorithm
# TODO: Implement loop closure
# TODO: Remove false/duplicate landmark locations
# TODO: Better copying landmarks when resampling


# Add noise to vector (vec) with given standard deviation (std)
def add_noise(vec, std):
    new = []
    for i in range(len(vec)):
        newVal = np.random.normal(vec[i], std)
        new.append(newVal)
    return new

# Landmark locations
LANDMARK_POSITIONS = np.array([[10.0, -2.0, 0],
              [15.0, 10.0, 1],
              [15.0, 15.0, 2],
              [10.0, 20.0, 3],
              [3.0, 15.0, 4],
              [-5.0, 20.0, 5],
              [-5.0, 5.0, 6],
              [-10.0, 15.0, 7],
              [22.0, 10, 8],
              [20.0, -4.0, 9],
              [5, -3.0, 10],
              [29.0, 20.0, 11],
              [14.0, 8.0, 12],
              [-10.0, 7.0, 13],
              ])

# Number of particles
N = 30

# Time step
t = 0.01

# Simulated noise standard deviation of observation
STANDARD_DEVIATION_NOISE_D = 0.05
STANDARD_DEVIATION_NOISE_THETA = np.deg2rad(1)

# Measurement covariance
Q = np.diag([STANDARD_DEVIATION_NOISE_D * 3, STANDARD_DEVIATION_NOISE_THETA * 3]) ** 2

# Standard deviation of gaussian for resampling
STANDARD_DEVIATION_XY = 0.5 / 2
STANDARD_DEVIATION_THETA = np.deg2rad(5)

# Predict poses noise
STANDARD_DEVIATION_PREDICT_XY = 0.01
STANDARD_DEVIATION_PREDICT_THETA = np.deg2rad(1)

# Resampling max weight limit
RESAMPLING_MAX_LIMIT = 0.01
# Minimum weight of particle
W_MIN = 0.01

# True odometry - [Velocity, Change in angle (Radians)]
u_true = [10, 1]

# Noisy odometry
#u_noise = [10.5, 0.7]
u_noise = add_noise([u_true[0]], 1) + add_noise([u_true[1]], np.deg2rad(5))

# Limits for graph
X_LIM = [-30, 30]
Y_LIM = [-30, 50]

def graph_main(lmks, u_true, Q): 
    # Initialize particles
    particles = [Particle(0, 0, 1 / N, 0) for _ in range(N)]

    # Initialize time
    time = 0
    
    # Initialize true location
    trueLoc =  [0, 0, 0]
    # Add to data history
    trueData = trueLoc

    # Initialize Location with noise
    noisyLoc = [0, 0, 0]
    # Add to data history
    noisyData = noisyLoc

    # Initialize predicted data history
    predictedData = [0, 0, 0]

    while True:        
        plt.cla()
        ax = plt.subplot()
        
        # Go forward one time step
        time += t

        # Calculate true location
        trueLoc = motion_model(trueLoc, u_true)

        # Calculate observation (Noisy)
        z = observation(lmks, trueLoc)

        # Call SLAM
        particles = FASTSLAM(z, u_noise, Q, particles)

        # Get array of particle locations
        particleLocs = [[particle.x, particle.y] for particle in particles]

        # Get height weight particle
        highestWeight = get_heightest_wegit(particles)

        # propagate test data
        noisyLoc = motion_model(noisyLoc, u_noise)

        # Add location data to respective data histories
        trueData = np.column_stack((trueData, np.transpose(trueLoc)))
        noisyData = np.column_stack((noisyData, np.transpose(noisyLoc)))
        predictedData = np.column_stack((predictedData, np.transpose([highestWeight.x, highestWeight.y, 0])))

        # Plot true landmarks (Black stars)
        for lmk in LANDMARK_POSITIONS:
            plt.plot(lmk[0], lmk[1], "*k")


        # Plot particles (Grey triangles)
        for loc in particleLocs:
            plt.plot(loc[0], loc[1], marker="^", markeredgecolor="grey", markerfacecolor="grey")

        # Get highest weight particle location
        x = highestWeight.x
        y = highestWeight.y
        theta = highestWeight.theta

        # Get average particle location
        # x = sum([particle.x for particle in particles]) / N
        # y = sum([particle.y for particle in particles]) / N
        # theta = sum([particle.theta for particle in particles]) / N
        
        # Calculate end point location of heading line
        ang_x = x + 2 * math.cos(theta)
        ang_y = y + 2 * math.sin(theta)

        # Plot hightest weight / average particle location (Purple triangle)
        plt.plot(x, y, "^m")
        # Plot heading line (Purple line)
        plt.plot([x, ang_x], [y, ang_y], "m")

        # Get predicted landmark locations of highest weight particle
        predicted_landmarks = highestWeight.get_nodes(highestWeight.lm)
        for lmk in predicted_landmarks:
            plt.plot(lmk.x, lmk.y, "*r")

            # Calculate and plotting covariance ellipse (Blue circle - 1 standard deviation)
            lambda_, v = np.linalg.eig(lmk.covariance)

            # make sure covariance is valid
            assert lambda_[0] > 0 and lambda_[1] > 0

            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(lmk.x, lmk.y),
            width=lambda_[0]*2, height=lambda_[1]*2,
            angle=np.rad2deg(np.arccos(v[0, 0])))
            ell.set_facecolor('blue')
            ax.add_artist(ell)

        # Plot noisy lidar measurements (Blue star)
        trueLocParticle = Particle(trueLoc[0], trueLoc[1], 0, trueLoc[2])
        for lmk in z:
            xy = getxy(lmk, trueLocParticle)
            plt.plot(trueLocParticle.x + xy[0], trueLocParticle.y + xy[1], "*b")



        # Plot predicted location data history (Purple line)
        plt.plot(predictedData[0, :],predictedData[1, :], "-m")
        # Plot noisy location data history (Purple line)
        plt.plot(noisyData[0, :],noisyData[1, :], "-b")
        # Plot true location data history (Purple line)
        plt.plot(trueData[0, :],trueData[1, :], "-k")

        # Plot current time 
        plt.title("Time: " + str(round(time, 2)))

        # Plot setup
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(X_LIM)
        plt.ylim(Y_LIM)
        plt.autoscale(False)

        # Pause graph for viewing
        plt.pause(0.001)
                
# Takes in observation (z), u (odometry) and measurement covariance (Q)
def main(z, u, Q, particles):
    particles = [Particle(0, 0, 1 / N, 0) for _ in range(N)]

    while True:
        particles = FASTSLAM(z, u, Q, particles)


# z - all observations, u - odometry input
def FASTSLAM(z, u, Q, particles):  

    # Propagate particles
    particles = predict_poses(particles, u)

    # Update particles
    for lmk_obs in z:
        lmk_val = lmk_obs[2]
        for particle in particles:
            lmk = particle.find(lmk_val) 
            # If landmark has not been seen
            if lmk == False:
               add_landmark(Q, lmk_obs, particle)
            # If landmark seen before
            else:
                update_landmark(Q, lmk, lmk_obs, particle)
    
    # Resample particles
    particles = resample_particles(particles, N, W_MIN)

    return particles 

# Calculate simulated observation based on landmark locations
def observation(lmkLocs, currLoc):
    z = []
    for lmk in lmkLocs:
        dx = lmk[0]-currLoc[0]
        dy = lmk[1]-currLoc[1]
        
        # Calculate distance and theta
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = pi_angle(math.atan2(dy, dx) - currLoc[2])


        # Add simulated noise to distance and theta
        d = add_noise([d], STANDARD_DEVIATION_NOISE_D)[0]
        theta = pi_angle(add_noise([theta], STANDARD_DEVIATION_NOISE_THETA)[0])
        
        loc = [d, theta, lmk[2]]

        z += [loc]
    return z

# Propagate particles based on odometry
# u = odometry measurements
def predict_poses(particles, u):
    for particle in particles:
        current = np.zeros((3))
        current[0] = particle.x
        current[1] = particle.y
        current[2] = particle.theta

        # Add noise to add variance to particles
        u_with_noise = add_noise([u[0]], STANDARD_DEVIATION_PREDICT_XY) + add_noise([u[1]], STANDARD_DEVIATION_PREDICT_THETA)

        new = motion_model(current, u_with_noise)
        particle.x = new[0]
        particle.y = new[1]
        particle.theta = new[2]
    return particles

# Adds new landmark
def add_landmark(Q, z, particle):
    dx, dy, val = getxy(z, particle)
    # Get location
    x, y, val = particle.x + dx, particle.y + dy, val
    # Calculate Jacobian
    jacobian = compute_jacobian(particle, [x, y])
    # Get covariance
    covariance = init_covariance(jacobian, Q)
    # Insert landmark for particle
    particle.insert(val, x, y, covariance)

# Updates existing landmark
def update_landmark(Q, lmk, z, particle):
    # Find the expected observation for the landmark
    x_k = lmk.x - particle.x
    y_k = lmk.y - particle.y
    d = math.sqrt(x_k ** 2 + y_k ** 2)
    theta = pi_angle(math.atan2(y_k, x_k) - particle.theta)
    # Expected observation
    z_k = [d, theta]

    # Update landmark location and covariance
    lmk.x, lmk.y, lmk.covariance, Q = ekf_update(lmk, z, z_k, particle, Q)
    # Update landmark weight 
    particle.w *= calculate_weight(Q, [z[0], z[1]], z_k)

# Sample Proposal Func
# particles is list of particles
# w_min is the threshold value for which particles will be discarded
def resample_particles(particles, num_particles, w_min):
    # TODO: this could be a poor resampling procedure, may need to find/choose a better one

    # create an empty list to hold the particles that we will return
    nu_particles = []

    particles_deleted = 0

    # Normalize weights of particles
    particles = normalize_weights(particles)

    # Backup particle incase all particles are deleted
    heightest_wegit_particle = get_heightest_wegit(particles)

    # Loop through all particles
    for i in range(0, num_particles):

        # Check that particles is greater than min threshold and sample with random probability
        if particles[i].w > w_min and particles[i].w > np.random.random() * RESAMPLING_MAX_LIMIT:
            nu_particles.append(particles[i])
        else:
            particles_deleted += 1
    
    # Get new highest weight particle
    if len(nu_particles) > 0:
        heightest_wegit_particle = get_heightest_wegit(nu_particles)
    
    # Replenish deleted particles based on highest weight particle
    x, y, theta = heightest_wegit_particle.x, heightest_wegit_particle.y, heightest_wegit_particle.theta
    for i in range(0, particles_deleted):
        x_new = np.random.normal(x, STANDARD_DEVIATION_XY)
        y_new = np.random.normal(y, STANDARD_DEVIATION_XY)
        theta_new = np.random.normal(theta, STANDARD_DEVIATION_THETA)
        replacement_particle = Particle(x_new, y_new, 1/num_particles, theta_new)

        replacement_particle.lm = heightest_wegit_particle.copytree()

        nu_particles.append(replacement_particle)

    return normalize_weights(nu_particles)

# Performs EKF update for a singular landmark
# input: landmark, actual observation, expected observation, particle, measurement covariance
# output: new landmark vector and landmark covariance
def ekf_update(lm, z_t, z_k, particle, Q): 
    z_t = z_t[:2]
    lm_vec = [lm.x, lm.y]
    lm_cov = lm.covariance
    H = compute_jacobian(particle, lm_vec) # Calculate jacobian
    Q = H @ lm_cov @ np.transpose(H) + Q # Calculate covariance
    K = calculate_kalman_gain(lm_cov, H, Q) # Calculate Kalman gain
    lm_vec = lm_vec + K @ (np.subtract(z_t, z_k)) # Calculate landmark vector
    lm_cov = (np.eye(2) - (K @ H)) @ lm_cov # Calculate landmark covariance
    return lm_vec[0], lm_vec[1], lm_cov, Q

# Calculate location of particle based on odometry data
# input: pose-[x, y, theta], u-[velocity, change in theta]
# output: new x array for new vehicle state
def motion_model(x, u):
    vel, change_theta = u[0], u[1]
    new_theta = change_theta * t + x[2]
    
    model = np.array([vel * t * math.cos(new_theta),
                  u[0] * t * math.sin(new_theta),
                  t * change_theta])
    
    new_x = x + model
    new_x[2] = pi_angle(new_x[2])

    return new_x

# Initialize Covariance
def init_covariance(H, Q):
    # Inverse of jacbobian
    H_inv = np.linalg.inv(H)
    # Transpose of inverse
    H_invT = np.transpose(H_inv)
    # Return the initialized covariance
    return H_inv @ Q @ H_invT

# Calculates Kalman gain
# Inputs: H - Jacobian, Q - measurement covariance
def calculate_kalman_gain(lm_cov, H, Q):
    # Get measurement matrix inverse
    Q_inverse = np.linalg.inv(Q)
    # Return gain using Kalman gain formula
    return lm_cov @ np.transpose(H) @ Q_inverse

# Calculate the jacobian
def compute_jacobian(particle, lm_vec):
    # lm_vec is landmark position vector
    # get delta x and delta y
    dx = lm_vec[0] - particle.x
    dy = lm_vec[1] - particle.y
    d_sqr = dx ** 2 + dy ** 2
    d = math.sqrt(d_sqr)

    #predicted z in polar coordinates
    # zp = np.array([d, pi_angle(math.atan2(dy, dx) - particle.theta)]).reshape(2, 1)

    # Jacobian formula
    Hf = np.array([[dx / d, dy / d], [-dy / d_sqr, dx / d_sqr]])

    return Hf

# Calculates Weights
# Q: Measurement covariance
# z_t: Actual observation
# z_k: Expected observation
def calculate_weight(Q, z_t, z_k):
    diff = np.subtract(z_t, z_k)
    w = ( np.linalg.det(np.multiply(2 * np.pi, Q)) ** -0.5 ) * ( np.exp((-0.5) * np.transpose(diff) @ np.linalg.inv(Q) @ diff) )
    return w

# Ensure that result is between -pi and pi
def pi_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

# Convert observation into coordinate position
def getxy(obs, particle):
    d = obs[0]
    theta = pi_angle(particle.theta + obs[1])

    dx = d * math.cos(theta)
    dy = d * math.sin(theta)

    return dx, dy, obs[2]

# Normalizes the weights
def normalize_weights(particles):
    total = 0
    # Get total weight
    for particle in particles:
        total += particle.w
    
    # Normalize weight from 0 - 1
    if total != 0:
        for particle in particles:
            particle.w = particle.w/total
    # Set to default weight if all weights are 0
    else:
        for particle in particles:
            particle.w = 1/len(particles)
    
    return particles

# Returns highest weight particle
def get_heightest_wegit(particles):
    return max(particles, key=lambda x : x.w)


if __name__ == "__main__":
    graph_main(LANDMARK_POSITIONS, u_true, Q)