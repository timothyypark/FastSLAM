# TODO

- Implement R-Tree
- Play around with resampling algorithm
- Implement loop closure
- Implement data association
- Remove false/duplicate landmark locations

# Overview
**Common Input Formats**

- Observation - [Distance (m), Angle (Radians), id]
- Odometry - [Velocity (m/s), Change in angle (Radians/s)]

# Data Structures

## Landmark
```python
class Landmark:
```
This class defines each landmark, which in our case is a cone on a track. Landmarks are stored in a binary search tree, where each instance of the landmark class is a node in the binary search tree. 

Nodes are organized based upon their value, which in this case is given. It also stores attributes for the x coordinate, y coordinate and covariance matrix of the cone's location. In addition, it also stores the node's left child, right child, and parent in the binary search tree.

## Landmark Constructors

```python
def __init__(self, val, x, y, covariance):
```
**Description:** \
Initializes attributes of the Landmark class.

**Parameters:**
- val - int - the ID of the landmark
- x - int - the landmark's x-coordinate
- y - int - the landmark's y-coordinate
- covariance - 2x2 numPy array - the covariance matrix that represents the uncertainty in the cone's location

## Landmark Methods
<!--- Insert Method -->
```python
def insert(self, val, x, y, covariance):
```
**Description:** \
Inserts a new landmark node into the binary search tree based upon the landmark ID.

**Parameters:**
- val - int - the ID of the landmark
- x - int - the landmark's x-coordinate
- y - int - the landmark's y-coordinate
- covariance - 2x2 numPy array - the covariance matrix that represents the uncertainty in the cone's location

**Return Value:** \
None


&nbsp;
<!--- deleteNode Method -->

```python
def deleteNode(self, key):
```
**Description:** \
Deletes the landmark node in the binary search tree with the corresponding landmark ID and balances the tree.

**Parameters:**
- key - int - the landmark ID of the node that will be deleted

**Return Value:** \
None 


&nbsp;  
<!--- deleteRoot Method -->

```python
def deleteRoot(self):
```
**Description:** \
Deletes the landmark node that is the root of the tree and balances the tree.

**Parameters:** \
None

**Return Value:** \
Root node of the new landmark binary search tree

&nbsp;  
<!--- find Method -->

```python
def find(self, key):
```
**Description:** \
Finds and returns the node with the desired landmark ID. If it does not exist, returns False. 

**Parameters:** 
- key - int - the landmark ID of the desired landmark

**Return Value:** \
If the desired node exists, it is returned. If it does not exist, the function returns False

&nbsp;  
<!--- storeBSTNodes Method -->

```python
def storeBSTNodes(self, root, nodes):
```
**Description:** \
Takes a tree and creates a list of its nodes recursively, inputted in inorder. 

**Parameters:**
- root - Landmark Object - the root of the Landmark Tree that is being made into a list
- nodes - List - a list where the nodes will be inserted into

**Return Value:** \
None

&nbsp;  
<!--- balanceUtil Method -->

```python
def balanceUtil(self, list_of_nodes, start, end):
```
**Description:** \
Constructs a binary search tree out of a list of nodes recursively. 

**Parameters:**
- list_of_nodes - List - a list where the nodes to be inserted into the tree
- start - int - the lowest indexed node in list_of_nodes being inserted into the tree
- end - int - the highest indexed node in list_of_nodes being inserted into the tree

**Return Value:** \
Newly-balanced landmark binary search tree

&nbsp;  
<!--- balance Method -->

```python
def balance(self, list_of_nodes):
```
**Description:** \
Constructs a balanced landmark binary search tree using balanceUtil.

**Parameters:**
- list_of_nodes - List - a list where the nodes to be inserted into the tree

**Return Value:** \
Newly-balanced landmark binary search tree (from balanceUtil)

&nbsp;  
<!--- preorder Method -->

```python
def preorder(self):
```
**Description:** \
Prints landmark IDs for the preorder traversal of the binary search tree.

**Parameters:** \
None

**Return Value:** \
None

&nbsp;  
<!--- inorder Method -->

```python
def inorder(self):
```
**Description:** \
Prints landmark IDs for the inorder traversal of the binary search tree.

**Parameters:** \
None

**Return Value:** \
None

&nbsp;  
<!--- inorder Method -->

```python
def postorder(self):
```
**Description:** \
Prints landmark IDs for the postorder traversal of the binary search tree.

**Parameters:** \
None

**Return Value:** \
None

&nbsp;  
<!--- copytree Method -->

```python
def copytree(self):
```
**Description:** \
Creates a copy of the entire landmark tree.

**Parameters:** \
None

**Return Value:** \
Root of the copied version of landmark tree.

## Particle
```python
class Particle:
```
This class defines each particle, which is a single proposal of the car's location, orientation, and surrounding landmarks.

Each particle stores attributes for its x-coordinate, y-coordinate, weight, heading, and the root of its landmark tree.

## Particle Constructors
```python
def __init__(self, x, y, w, theta):
```
**Description:** \
Initializes attributes of the Particle class.

**Parameters:**
- x - int - the particle's x-coordinate
- y - int - the particle's y-coordinate
- w - int - the particle's weight
- theta - int - the particle's heading

## Particle Methods
<!--- insert Method -->

```python
def insert(self, val, x, y, covariance):
```
**Description:** \
Inserts a new landmark into the particle's landmark binary search tree.

**Parameters:**
- val - int - the ID of the landmark
- x - int - the landmark's x-coordinate
- y - int - the landmark's y-coordinate
- covariance - 2x2 numPy array - the covariance matrix that represents the uncertainty in the cone's location

**Return Value:** \
None

&nbsp;
<!--- delete Method -->

```python
def delete(self, val):
```
**Description:** \
Deletes the desired node from the particle's landmark tree

**Parameters:**
- val - int - the landmark ID of the desired node to be deleted

**Return Value:** \
None

&nbsp;  
<!--- find Method -->

```python
def find(self, val):
```
**Description:** \
Finds and returns the node with the desired landmark ID in the particle's landmark tree. If it does not exist, returns False. 

**Parameters:** 
- val - int - the landmark ID of the desired landmark

**Return Value:** \
If the desired node exists, it is returned. If it does not exist, the function returns False

&nbsp;  
<!--- preorder Method -->

```python
def preorder(self):
```
**Description:** \
Prints landmark IDs for the preorder traversal of the particle's landmark tree.

**Parameters:** \
None

**Return Value:** \
None

&nbsp;  
<!--- inorder Method -->

```python
def inorder(self):
```
**Description:** \
Prints landmark IDs for the inorder traversal of the particle's landmark tree.

**Parameters:** \
None

**Return Value:** \
None

&nbsp;  
<!--- inorder Method -->

```python
def postorder(self):
```
**Description:** \
Prints landmark IDs for the postorder traversal of the particle's landmark tree.

**Parameters:** \
None

**Return Value:** \
None

&nbsp;  
<!--- copytree Method -->

```python
def copytree(self):
```
**Description:** \
Creates a copy of the particle's entire landmark tree.

**Parameters:** \
None

**Return Value:** \
Root of the copied version of particle's landmark tree.

# Methods

```python
def graph_main(lmks, u_true, Q)
```

**Description** \
Continuously run and plot SLAM

**Parameters**

- lmks - list - List of observations
- u_true - list - List of odometry
- Q - 2D array - Lidar covariance

**Returns** \
None

```python
def SLAMMEDADDY(z, u, Q, particles)
```

**Description** \
Runs one iteration of SLAM

**Parameters**

- z - list - List of observations
- u - list - List of odometry
- Q - 2D array - Lidar covariance
- Particles - List of Particles

**Returns** \
List of particles after the iteration of SLAM


```python
def observation(lmkLocs, currLoc)
```

**Description** \
Calculate simulated observation based on landmark locations

**Parameters**
- lmkLocs - list - List of landmarks in the form of [x(m), y(m), id]. Initial car position is (0, 0).
- currLoc - list - Current car location [x, y]

**Returns** \
List of observations

```python
def predict_poses(particles, u)
```

**Description** \
Propagate particles based on odometry. Adds noise to increase variance among particles.

**Parameters**
- particles - List of particles
- u - Odometry data 

**Returns** \
List of propagated particles based on odometry


```python
def add_landmark(Q, z, particle)
```

**Description** \
Adds a new landmark to the particle.

**Parameters**
- Q - 2D array - Lidar covariance
- z - observation of landmark
- particle - particle to add landmark to

**Returns** \
None

```python
def update_landmark(Q, lmk, z, particle)
```

**Description** \
Updates the given landmark and particle

**Parameters**
- Q - 2D array - Lidar covariance
- lmk - landmark to be changed
- z - observation of landmark
- particle - particle to be changed

**Returns** \
None

```python
def resample_particles(particles, num_particles, w_min)
```

**Description** \
Resample and normalize weights of particles

**Parameters**
- particles - list of particles
- num_particles - number of desired particles to be outputted
- w_min - minimum weight of a particle

**Returns** \
New list of particles with normalized weights

```python
def ekf_update(lm, z_t, z_k, particle, Q)
```

**Description** \
Performs EKF update for a singular landmark

**Parameters**
- lm - landmark to be changed
- z_t - actual observation
- z_k - expected observation
- particle - particle to be changed
- Q - lidar covariance

**Returns**
- new landmark x
- new landmark y
- new landark covariance
- new lidar covariance
  
```python
def motion_model(x, u)
```

**Description** \
Calculate location of particle based on odometry data

**Parameters**
- x - current vehicle pose - [x, y, theta]
- u - odometry data

**Returns**\
vehicle location after one time step
  
```python
def init_covariance(H, Q):
```

**Description** \
Returns the intial covariance of a landmark

**Parameters**
- H - Jacobian
- Q - Lidar covariance

**Returns**\
Intial covariance of landmark


```python
def calculate_kalman_gain(lm_cov, H, Q)
```

**Description** \
Calculates Kalman gain

**Parameters**
- lm_cov - landmark covariance
- H - Jacboain
- Q - Lidar covariance


**Returns**\
Kalman gain of landmark

```python
def compute_jacobian(particle, lm_vec)
```

**Description** \
Calculates Jacobian

**Parameters**
- particle - current particle
- lm_vec - current landmark postion [x, y]

**Returns**\
Jacboain matrix of landmark

```python
def calculate_weight(Q, z_t, z_k)
```

**Description** \
Calculates new weight

**Parameters**
- Q - Measurment covariance
- z_t - Actual observation
- z_k - Expected observation

**Returns**\
new weight

```python
def pi_angle(theta)
```

**Description** \
Ensure that result is between -pi and pi

**Parameters**
- theta - angle in radians

**Returns**\
new theta within -pi and pi

```python
def getxy(obs, particle)
```

**Description** \
Convert observation into coordinate position

**Parameters**
- obs - current observation
- particle - current particle

**Returns**\
location of landmark in xy coordinates

```python
def normalize_weights(particles)
```

**Description** \
Normalize particle wights

**Parameters**
- particles - list of particles

**Returns**\
particles with their weights normalized

```python
def get_heightest_wegit(particles)
```

**Description** \
Returns highest weight particle

**Parameters**
- particles - list of particles

**Returns**\
The highest weight particle

```python
def add_noise(vec, std)
```

**Description:** \
Adds noise each element using a normal distribution with the element as the mean and standard deviation std. Non-mutating. 

**Parameters:**

- vec - list - Elements containing original values
- std - float - Standard deviation of noise

**Returns** \
New 1D list with added noise
