### Test supporting data-structure functions
from DataStructs import Particle

def build_tree(lm):
    lm.insert(4,4,4,4)
    lm.insert(5,5,5,5)
    lm.insert(8,8,8,8)
    lm.insert(2,2,2,2)
    lm.insert(3,3,3,3)
    lm.insert(1,1,1,1)
    lm.insert(7,7,7,7)
    lm.insert(6,6,6,6)
    lm.insert(9,9,9,9)
    tree_orders(lm)

def tree_orders(tree):
    print('Inorder: ')
    tree.inorder()
    print('\n')
    """print('Preorder: ')
    tree.preorder()
    print('\n')
    print('PostOrder: ')
    tree.postorder()
    print('\n')"""


def test_bst_delete(lm, bad_node):
    print('Tree Inorder prior to deletion: ')
    tree_orders(lm)
    print('Node being deleted: ', bad_node)
    lm.delete(bad_node)
    print('Tree Inorder after deletion: ')
    tree_orders(lm)
    return

def run_bst_delete_tests(particle):
    print('Running tests for .delete function: ...')

    test_bst_delete(particle, 9)

    test_bst_delete(particle, 5)
    #delete node 3
    test_bst_delete(particle, 3)

    test_bst_delete(particle, 4)

def test_bst_find(lm, good_node):
    a = lm.find(good_node)
    b = good_node
    if a and a.val==b:
        print('Node ' + str(good_node) + ' was found')
    else:
        print('Node ' + str(good_node) + ' was not found')

def run_bst_find_tests(particle):
    print('Running BST Tests for .find function: ...')

    test_bst_find(particle, 8)

    test_bst_find(particle, 10)

    test_bst_find(particle, 1)

    test_bst_find(particle, 5)
    print('\n')

def test_copytree(particle):
    print("Copying Tree")

    new_p = particle.copytree()
    new_p.inorder()
    if new_p is not particle.lm:
        ppp = True
    else:
        ppp = False
    print("new_tree is not the same as original tree: " + str(ppp))

    new_p.deleteNode(8)

    new_p.inorder()
    particle.inorder()
    print()

def run_tests():

    first_particle = Particle(0,0,100,0.5)

    build_tree(first_particle)

    run_bst_find_tests(first_particle)
    #delete node 4
    run_bst_delete_tests(first_particle)

    test_copytree(first_particle)
run_tests()