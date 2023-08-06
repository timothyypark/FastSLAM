
#import numpy as np
# Notes for improvement:

class Landmark:
    # Defines landmarks as storing values for index and location, as well as covariance and nodes below it
    # Value was intended to be used to place node in correct location on BST, but can be changed as well

    def __init__(self, val, x, y, covariance):
        self.val = val # marker for which landmark it is/ which measurement
        self.x = x
        self.y = y # location of this observation of the landmark
        self.covariance = covariance # covariance used for EKF
        self.left = None
        self.right = None # Child nodes
        self.parent = None
        

    """Getters and Setters"""
    def getVal(self):
        return self.val
    def setVal(self, val):
        self.val = val
    def getX(self):
        return self.x
    def setX(self, x):
        self.x = x
    def getY(self):
        return self.y
    def setY(self, y):
        self.y = y
    def getCovariance(self):
        return self.covariance
    def setCovariance(self, covariance):
        self.covariance = covariance
    def getLeft(self):
        return self.left
    def setLeft(self, left):
        self.left = left
    def getRight(self):
        return self.right
    def setRight(self, right):
        self.right = right
    def getParent(self):
        return self.parent
    def setParent(self, parent):
        self.parent = parent

    # Defines method for inserting new value as new node in BST
    # Input: root node(current node), value to be inserted, parent node
    # Note: As the function recurses, root becomes the current node, and parent becomes the parent of the current node
    def insert(self, val, x, y, covariance):
        if self.val == val:
            return False        # As BST cannot contain duplicate data

        elif val < self.val:
            # Data less than the root data is placed to the left of the root
            if self.left:
                return self.left.insert(val, x, y, covariance)
            else:
                self.left = Landmark(val, x, y, covariance)
                self.left.parent = self

        else:
            # Data greater than the root data is placed to the right of the root
            if self.right:
                return self.right.insert(val, x, y, covariance)
            else:
                self.right = Landmark(val, x, y, covariance)
                self.right.parent = self
    
    # deletes the node that has the key in the tree
    # inputs: key - int value representing the val of the node to be deleted
    def deleteNode(self, key):
        # Finds the node to be deleted
        bad_node = self.find(key)

        # puts all the nodes to the left of the bad node in a list
        
        subtree_left = bad_node.left
        subtree_left_nodes = []
        self.storeBSTNodes(subtree_left, subtree_left_nodes)

        # puts all the nodes to the right of the bad node in a list
        subtree_right = bad_node.right
        subtree_right_nodes = []
        self.storeBSTNodes(subtree_right, subtree_right_nodes)

        # puts this list together
        # subtree nodes is a list of all the nodes that were below the deleted node
        subtree_nodes = subtree_left_nodes + subtree_right_nodes

        #parent value and deleted value are used to point to where the branch being deleted was
        parent_value = bad_node.parent.val
        deleted_value = bad_node.val
        
        # if the node being deleted is a leaf, just make the parent's child equal to none
        if subtree_nodes:

            # sets the parent's child based upon the parent's relation to the child
            # can either be left or right
            if deleted_value > parent_value:
                bad_node.parent.right = self.balance(subtree_nodes)
            elif deleted_value < parent_value:
                bad_node.parent.left = self.balance(subtree_nodes)
        else:
            # same thing as with the subtree, but this time the parent's child gets evaluated to none
            if deleted_value > parent_value:
                bad_node.parent.right = None
            elif deleted_value < parent_value:
                bad_node.parent.left = None
        return
        
    # used to make a new tree if the root is the node being deleted
    # this function is called by the particle class, which is why it has no args
    def deleteRoot(self):
        
        # all nodes to the left of the root are put in a list
        subtree_left = self.left
        subtree_left_nodes = []
        self.storeBSTNodes(subtree_left, subtree_left_nodes)

        # all nodes to the right of the tree are put in a list
        subtree_right = self.right
        subtree_right_nodes = []
        self.storeBSTNodes(subtree_right, subtree_right_nodes)

        # subtree list now contains all the nodes in a single list
        subtree_nodes = subtree_left_nodes + subtree_right_nodes

        # creates the new tree by making it out of the list of trees
        root = self.balance(subtree_nodes)
        return root


    # Finds and returns node with the value asked for in key
    # inputs: key - int value representing the val of the node being found
    def find(self, key):

        # Base case: if key is found, return the node that it corresponds to
        if(key == self.val):
            return self
        
        # recursive case #1: if key is less than the val of the current node, evaluate the left child
        # if left child does not exist, return false
        elif(key < self.val):
            if self.left:
                return self.left.find(key)
            else:
                return False
            
        # recursive case #2: if key is less than the val of the current node, evaluate the right child
        # if right child does not exist, return false
        else:
            if self.right:
                return self.right.find(key)
            else:
                return False

    # takes the tree and puts the tree into a list
    # used as a helper function before balancing
    def storeBSTNodes(self, root, nodes):
        
        # Base case
        if not root:
            return
        
        # Store nodes in Inorder (which is sorted
        # order for BST)
        self.storeBSTNodes(root.left, nodes)
        nodes.append(root)
        self.storeBSTNodes(root.right, nodes)
    
    # Recursive function to construct binary tree
    # called by balance
    def balanceUtil(self, list_of_nodes, start, end):
        
        # base case
        if start>end:
            return None
    
        # Get the middle element and make it root
        mid=(start+end)//2
        node=list_of_nodes[mid]
    
        # Using index in Inorder traversal, construct
        # left and right subtress
        node.left=self.balanceUtil(list_of_nodes,start,mid-1)
        if node.left: # Unless this node is the child, initialize the current node as the child's parent
            node.left.parent = node
        node.right=self.balanceUtil(list_of_nodes,mid+1,end)
        if node.right: # Same as with the left child
            node.right.parent = node
        return node
    
    # This functions converts an unbalanced BST to
    # a balanced BST
    def balance(self, list_of_nodes):
        # Constructs BST from nodes[]
        n=len(list_of_nodes)
        return self.balanceUtil(list_of_nodes,0,n-1)
    
    
    # For preorder traversal of the BST
    def preorder(self):
        
        if self:
            print(str(self.val), end = ' ')
            if self.left:
                self.left.preorder()
            if self.right:
                self.right.preorder()

    # For Inorder traversal of the BST
    def inorder(self):
        
        if self:
            if self.left:
                self.left.inorder()
            print(str(self.val), end = ' ')
            if self.right:
                self.right.inorder()

    # For postorder traversal of the BST
    def postorder(self):
        
        if self:
            if self.left:
                self.left.postorder()
            if self.right:
                self.right.postorder()
            print(str(self.val), end = ' ')

    # for making a copy of the existing tree
    def copytree(self):

        copy_val = self.val
        copy_x = self.x
        copy_y = self.y
        copy_covariance = self.covariance

        # make copy of node
        root_copy = Landmark(copy_val, copy_x, copy_y, copy_covariance)

        # traverse the tree recursively
        if self.left:
            root_copy.left = self.left.copytree()
            root_copy.left.parent = root_copy
        if self.right:
            root_copy.right = self.right.copytree()
            root_copy.right.parent = root_copy
        
        return root_copy

class Particle:
    # Particle for vehicle location and heading (if applicable)
    def __init__(self, x, y, w, theta):
        self.x = x
        self.y = y # location of particle
        self.w = w # weight of particle
        self.theta = theta # heading of vehicle

        self.lm = None # root of BST of Landmarks
    
    def __str__(self) -> str:
        print(f"x: {self.x} \n" + 
              f"y: {self.y} \n" +
              f"w: {self.w} \n" +
              f"theta: {self.theta} \n")
    
    # Inserts a landmark into the tree before balancing
    # if root does not exist, root is initialized to the node being passed in
    def insert(self, val, x, y, covariance):
        # if the tree has been made already
        if self.lm is not None:
            self.lm.insert(val, x, y, covariance) # insert
            # balance function
            nodes = []
            self.lm.storeBSTNodes(self.lm, nodes)
            self.lm = self.lm.balance(nodes)

        # if there is no tree yet
        else:
            self.lm = Landmark(val, x, y, covariance)
    
    # Deletes a node with a given val from the tree
    def delete(self, val):
        if self.lm is not None:
            # if the node being deleted is the root node
            if self.lm.val == val:
                self.lm = self.lm.deleteRoot()
                return
            
            # for deleting all other nodes
            else:
                return self.lm.deleteNode(val)
        
    def find(self, val):
        if self.lm:
            return self.lm.find(val)
        else:
            return False
        
    def preorder(self):
        if self.lm is not None:
            print()
            self.lm.preorder()

    def inorder(self):
        print()
        if self.lm is not None:
            self.lm.inorder()

    def postorder(self):
        print()
        if self.lm is not None:
            self.lm.postorder()

    def copytree(self):
        if self.lm is not None:
            new_tree = self.lm.copytree()
            return new_tree



    """Getters and Setters"""
    def get_lm_Val(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.val
    def setVal(self, val):
        if self.lm:
            landmark = self.lm.find(val)
            self.lmval = val
    def getX(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.x
    def setX(self, x):
        self.x = x
    def getY(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.y
    def setY(self, y):
        self.y = y
    def getCovariance(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.covariance
    def setCovariance(self, covariance):
        self.covariance = covariance
    def getLeft(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.left
    def setLeft(self, left):
        self.left = left
    def getRight(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.key
    def setRight(self, right):
        self.right = right
    def getParent(self, key):
        if self.lm:
            landmark = self.lm.find(key)
            if landmark:
                return landmark.right
    def setParent(self, parent):
        self.parent = parent

    def get_nodes(self, lm):
        lmks = []
        if lm is None:
            return []
        lmks.append(lm)
        lmks.extend(self.get_nodes(lm.left))
        lmks.extend(self.get_nodes(lm.right))
        return lmks
    