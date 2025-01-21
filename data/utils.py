from openfold.utils import rigid_utils as ru


# Global map from chain characters to integers.
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def to_numpy(x):
    return x.detach().cpu().numpy()

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return ru.Rigid(rots=rots, trans=trans)
