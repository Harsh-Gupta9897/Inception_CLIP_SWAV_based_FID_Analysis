import numpy as np
import scipy.linalg


def compute_fid(real_features: np.ndarray,
                gen_features: np.ndarray) -> float:
    """Computes the Fréchet Inception Distance."""
    assert real_features.ndim == 2 and gen_features.ndim == 2
    assert real_features.shape[0] == gen_features.shape[0]

    # Feature statistics.
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # FID.
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return np.absolute(fid)



if __name__=='__main__':
    real_features,gen_features = None, None
    compute_fid(real_features,gen_features)