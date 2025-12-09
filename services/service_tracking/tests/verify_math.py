import numpy as np

def check_logic():
    N = 10
    means = np.random.rand(N, 8)
    covs = np.random.rand(N, 8, 8)
    S = np.eye(8) * 2 # Scale by 2
    T = np.ones(8)
    
    # Test Means logic
    # means @ S + T
    # (N,8) @ (8,8) -> (N,8). + (8,) -> (N,8)
    try:
        new_means = means @ S + T
        print(f"Means update shape: {new_means.shape}")
        # Verify value for first row
        expected_row0 = means[0] * 2 + 1
        if np.allclose(new_means[0], expected_row0):
            print("Means update logic CORRECT")
        else:
            print("Means update logic INCORRECT")
    except Exception as e:
        print(f"Means update FAILED: {e}")

    # Test Covariance logic
    # S @ covs @ S.T
    # (8,8) @ (N,8,8) -> ?
    try:
        new_covs = S @ covs @ S.T
        print(f"Covs update shape: {new_covs.shape}")
        
        # Manually compute single item
        # S * C[0] * S.T
        expected_cov0 = S @ covs[0] @ S.T
        if np.allclose(new_covs[0], expected_cov0):
             print("Covs update logic CORRECT")
        else:
             print("Covs update logic INCORRECT")
             
    except Exception as e:
        print(f"Covs update FAILED: {e}")

if __name__ == "__main__":
    check_logic()
