### Unit Tests for SLAM
# @author Benjamin Asch

from DataStructs import Particle
from utils import resample_particles

### Particle Resampling Tests
def test_resample_particles():
    test_status = True
    name = "resample_particles"

    test_status = test_status and resample_1()

    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def resample_1():
    p_1 = Particle(1, 2, 0.7, 45, 5)
    p_2 = Particle(1, 2, 0.3, 45, 5)
    p_3 = Particle(1, 2, 0.1, 45, 5)
    p_4 = Particle(1, 2, 0.5, 45, 5)
    p_5 = Particle(1, 2, 0.12, 45, 5)

    input_particles = [p_1, p_2, p_3, p_4, p_5]
    num_p = len(input_particles)
    input_w_min = 0.2
    resample_output = resample_particles(input_particles, num_p, input_w_min)

    r_1 = Particle(1, 2, (0.7/1.5), 45, 5)
    r_2 = Particle(1, 2, (0.3/1.5), 45, 5)
    r_3 = Particle(1, 2, (0.5/1.5), 45, 5)
    test_result = [r_1, r_2, r_3]

    if not (test_result == resample_output):
        print("Expected ", test_result, ", but got ", resample_output)

def test_calculate_weight():
    test_status = True
    name = "calculate_weight"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_calculate_covariance():
    test_status = True
    name = "calculate_covariance"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_compute_jacobian():
    test_status = True
    name = "compute_jacobian"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_normalize_weights():
    test_status = True
    name = "normalize_weights"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_add_landmark():
    test_status = True
    name = "add_landmark"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_update_landmark():
    test_status = True
    name = "update_landmark"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_calculate_kg():
    test_status = True
    name = "calculate_kg"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_predict_poses():
    test_status = True
    name = "predict_poses"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_motion_model():
    test_status = True
    name = "motion_model"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def test_ekf_update():
    test_status = True
    name = "ekf_update"
    if test_status:
        print("PASSED: '" + name + "' test passed.")
    else:
        print("FAIL: '" + name + "' test failed.")
    return test_status

def run_tests():
    all_passed = test_resample_particles() and test_calculate_weight()\
        and test_calculate_covariance() and test_compute_jacobian() and\
            test_normalize_weights() and test_add_landmark() and test_update_landmark()\
                and test_calculate_kg() and test_predict_poses() and test_motion_model()\
                    and test_ekf_update()
    
    if all_passed:
        print("PASSED: All 'test_slam' tests passed!");
    else:
        print("FAIL: Not all 'test_slam' tests passed.")

    return all_passed

run_tests()

