from sklearn.utils import Bunch
from nilearn.datasets import load_mni152_brain_mask, fetch_atlas_juelich as fetch_atlas_juelich_nilearn, fetch_atlas_aal as fetch_atlas_aal_nilearn, fetch_atlas_harvard_oxford as fetch_atlas_harvard_oxford_nilearn
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.image import resample_img
from nibabel.affines import apply_affine
import os
import numpy as np
import pandas as pd
import nibabel as nib
import hashlib
import json
import logging
from typing import List, Optional, Tuple, Union
from sqlalchemy.exc import NoSuchTableError
from .models_sqlalchemy_orm import *
from sqlalchemy.orm import Session as _Session
from sqlalchemy import  and_, select
import warnings
import gzip
from io import BytesIO
# from django.core.files.storage import default_storage
from PIL import Image
from rapidfuzz import process, fuzz
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy.orm.exc import NoResultFound


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Random helper functions"""

def md5_hash(input_data: Union[str, bytes, object]) -> str:
    """
    Computes the MD5 hash of the input data.
    
    Args:
        input_data (Union[str, bytes, object]): The input data can be a file path (str), bytes object, 
                                                or an instance of a class that can be converted to bytes.
        
    Returns:
        str: The MD5 hash of the input data.

    Notes:
        - The MD5 of the in-memory data will be different from the MD5 of the file on disk. 
        - I am not aware of any good workaround for this.
    """
    md5 = hashlib.md5()
    
    if isinstance(input_data, str):
        with open(input_data, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
    elif isinstance(input_data, bytes):
        md5.update(input_data)
    elif hasattr(input_data, 'to_bytes'):
        md5.update(input_data.to_bytes())
    elif hasattr(input_data, 'get_data'):
        md5.update(input_data.get_data().tobytes())
    elif hasattr(input_data, 'get_fdata'):
        md5.update(input_data.get_fdata().tobytes())
    else:
        raise TypeError("Input data must be a string (file path), bytes object, or an instance with a 'to_bytes' or 'get_data' method.")
    
    return md5.hexdigest()

def determine_filetype(filepath: str) -> str:
    """
    Determines the filetype of a file based on its extension.
    """
    filetype_checks = {
        'nii.gz': lambda f: f.endswith('.nii.gz'),
        'nii': lambda f: '.nii' in f and not f.endswith('.nii.gz'),
        'npy': lambda f: f.endswith('.npy'),
        'npz': lambda f: f.endswith('.npz'),
        'gii': lambda f: f.endswith('.gii') or '.gii' in f,
        'mgz': lambda f: f.endswith('.mgz'),
        'surf': lambda f: any(s in f.lower() for s in ['lh.', 'rh.', '.surf']),
        'label': lambda f: '.label' in f.lower(),
        'annot': lambda f: '.annot' in f.lower(),
        'fsaverage': lambda f: 'fsaverage' in f.lower(),
        'freesurfer': lambda f: any(s in f.lower() for s in ['aparc', 'aseg', 'bert', 'curv', 'sulc', 'thickness']),
        'png': lambda f: f.endswith('.png'),
        'jpg': lambda f: f.endswith('.jpg') or f.endswith('.jpeg')
    }

    for filetype, check in filetype_checks.items():
        if check(filepath):
            return filetype
    
    return 'unknown'

# def fetch_from_s3(filepath):
#     extension = determine_filetype(filepath)

#     with default_storage.open(filepath) as file:
#         file_data = file.read()

#     if extension == 'nii.gz':
#         fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=BytesIO(file_data)))
#         return nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

#     elif extension == 'nii':
#         fh = nib.FileHolder(fileobj=BytesIO(file_data))
#         return nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

#     elif extension == 'npy':
#         return np.load(BytesIO(file_data), allow_pickle=True)

#     elif extension in ['png', 'jpg', 'jpeg']:
#         return Image.open(BytesIO(file_data))

#     else:
#         raise ValueError(f"Unsupported file type: {extension}")

"""Functions for manipulating imaging data itself"""

def fetch_2mm_mni152_mask(resolution=2):
    """Loads the MNI152 template in 2mm resolution with shape = (91, 109, 91)"""
    target_shape = (91, 109, 91)
    target_affine = np.array([[2, 0, 0, -90],
                              [0, 2, 0, -126],
                              [0, 0, 2, -72],
                              [0, 0, 0, 1]])
    return resample_img(
        load_mni152_brain_mask(resolution=resolution, threshold=0.10),
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation='nearest'
    )

def add_name_attribute(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            result.name = name
            return result
        return wrapper
    return decorator

@add_name_attribute('Harvard-Oxford Atlas')
def fetch_atlas_harvard_oxford(*args, **kwargs):
    return fetch_atlas_harvard_oxford_nilearn(*args, **kwargs)

@add_name_attribute('Juelich Atlas')
def fetch_atlas_juelich(*args, **kwargs):
    return fetch_atlas_juelich_nilearn(*args, **kwargs)

def fetch_atlas_biancardi_brainstem(data_dir='/Users/jt041/repos/db_engineering/biancardi_brainstem_atlas'):
    """
    Fetch the brainstem nuclei atlas and return as a Bunch object.
    
    Parameters:
    - data_dir: str, optional
        Path to the directory containing the atlas files.
    
    Returns:
    - Bunch object with 'maps', 'labels', and 'description'.
    """
    atlas_img_path = os.path.join(data_dir, 'brainstem_nuclei_atlas_2mm.nii.gz')
    labels_csv_path = os.path.join(data_dir, 'brainstem_nuclei_labels.csv')
    
    # Load the atlas image
    atlas_img = nib.load(atlas_img_path)
    
    # Load the labels
    labels_df = pd.read_csv(labels_csv_path)
    labels = labels_df['labels'].tolist()
    
    # Create and return the Bunch object
    atlas_bunch = Bunch(
        name='Biancardi Brainstem Nuclei Atlas',
        maps=atlas_img,
        labels=labels,
        description="Brainstem nuclei atlas resampled to 2mm MNI152 space.",
        filename=atlas_img_path
    )
    
    return atlas_bunch

def fetch_atlas_3209c91v():
    """Fetches the 3209c91v parcellation of the MNI152 brain."""
    path = os.path.join(os.path.dirname(__file__),'data/3209c91v.nii.gz')
    img = nib.load(path)
    unique_values = np.unique(img.get_fdata())
    unique_values = unique_values.astype(int).tolist()
    labels = ['Background' if value == 0 else f'Chunk {value}' for value in unique_values]
    description = """Parcellation of 2mm MNI152 brain voxels into 
                    3209 equally sized chunks of 91 voxels, 
                    courtesy of William Drew (MD/PhD Candidate at Columbia University)."""
    atlas_dict = {
        'name': '3209c91v',
        'filename': path,
        'maps': img,
        'labels': labels,
        'description': description
    }
    return Bunch(**atlas_dict)

def fetch_atlas_aal():
    """
    Fetches the AAL atlas from Nilearn, and modifies it to have consecutive indices.
    This step is necessary for compatibility with other atlas/parcellation schemes,
    and for using the NiftiLabelsMasker in Nilearn.
    """

    atlas = fetch_atlas_aal_nilearn()
    
    original_indices = [int(idx) for idx in atlas.indices]
    labels = atlas.labels
    consecutive_indices = list(range(1, len(original_indices) + 1))
    index_mapping = dict(zip(original_indices, consecutive_indices))

    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()

    modified_atlas_data = np.zeros_like(atlas_data)
    
    for original_idx, new_idx in index_mapping.items():
        modified_atlas_data[atlas_data == original_idx] = int(new_idx)

    modified_atlas_img = nib.Nifti1Image(modified_atlas_data.astype(np.int32), atlas_img.affine, atlas_img.header)

    modified_labels = ['Background'] + labels
    modified_indices = [0] + consecutive_indices

    modified_atlas = Bunch(
        name='AAL',
        maps=modified_atlas_img,
        labels=modified_labels,
        indices=modified_indices,
        description=atlas.description
    )

    return modified_atlas

def apply_parcellation(voxelwise_map, parcellation, strategy='mean', return_region_ids=False):
    """
    Uses nilearn to apply a parcellation to a brain map.
    Returns a 1d array of the parcel values.
    Can take the mean or sum of the data in each parcel (if both are true, it will take the mean).
    
    Params:
    - voxelwise_map: a 3d NIfTI image of the data.
    - parcellation: an sklearn bunch object with 'maps' and 'labels' attributes.
    - strategy: 'mean' or 'sum'
    Returns:
    - data: a 2d array of the data in each parcel of shape (n_samples, n_parcels)
    """    
    masker = NiftiLabelsMasker(labels_img=parcellation.maps, 
                               labels=parcellation.labels,
                               strategy=strategy)
    data = masker.fit_transform(voxelwise_map)
    if return_region_ids:
        region_ids = np.array(list(masker.region_ids_.values())[1:])
        return data, region_ids
    return data.astype(float)

"""SQL helper functions"""

def get_user_id(session: _Session) -> int:
    user = session.query(User).filter_by(username="josephturner").first()
    if not user:
        user = User(
            username="josephturner",
            email="jiturner@bwh.harvard.edu",
            password="joseph's fake password lol",
            first_name="Joseph",
            last_name="Turner",
            is_superuser=True,
            is_staff=True
        )
        session.add(user)
        session.commit()
    return user.id

def get_file_id(filepath, table, session):
    """
    Uses SQLAlchemy to get {map_type}_files.id where {map_type}_files.path = filepath.
    """
    result = session.query(table.id).filter(table.path == filepath).first()
    return result.id if result else None

def get_parcellation_id(parcellation_name, session):
    """
    Uses SQLAlchemy to get the id where parcellations.name = parcellation_name.
    """
    return session.query(Parcellation.id).filter(Parcellation.name == parcellation_name).scalar()

def get_parcel_id(parcellation_id, parcel_value, session):
    """
    Uses SQLAlchemy to get parcel.id where parcellations.id = parcellation_id AND parcels.value = parcel_value.
    """
    return session.query(Parcel.id).filter(Parcel.parcellation_id == parcellation_id, Parcel.value == parcel_value).scalar()

def get_labels_at_xyz(x: int, y: int, z: int, session: _Session) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Uses SQLAlchemy to get the labels of all parcels at the given MNI152 coordinates.
    Returns a list of tuples, where each tuple contains the label and the parcellation name.
    Note: Shouldn't there be a more efficient way to do this in one query? Regular SQL can do this with chained joins.
    """
    # Query to find all parcel_ids in VoxelwiseValue where the mni_x, mni_y, mni_z values match
    voxels = session.query(VoxelwiseValue).filter(
        and_(
            VoxelwiseValue.mni152_x == x,
            VoxelwiseValue.mni152_y == y,
            VoxelwiseValue.mni152_z == z
        )
    ).all()

    if not voxels:
        return []

    results = []

    for voxel in voxels:
        parcel = session.query(Parcel).filter(Parcel.id == voxel.parcel_id).first()
        if parcel is None:
            continue
        parcellation = session.query(Parcellation).filter(Parcellation.id == parcel.parcellation_id).first()
        if parcellation is None:
            continue
        results.append((parcel.label, parcellation.name))

    return results

def get_files_at_xyz(x: int, y: int, z: int, map_type: str, session: _Session) -> List[Tuple[str, str, float]]:
    """
    Uses SQLAlchemy to get the paths of all files of a given map type that contain data at the given MNI152 coordinates.
    Returns a list of tuples, where each tuple contains the file path, map type, and the value.
    
    Args:
        x (int): MNI152 x-coordinate.
        y (int): MNI152 y-coordinate.
        z (int): MNI152 z-coordinate.
        map_type (str): The type of map ('connectivity', 'roi', 'group_level_map').
        session (_Session): SQLAlchemy session object.
        
    Returns:
        List[Tuple[str, str, float]]: List of tuples containing file paths, map types, and values, ordered by descending value.
    """
    # Query to find all parcel_ids in VoxelwiseValue where the mni_x, mni_y, mni_z values match
    voxels = session.query(VoxelwiseValue).filter(
        and_(
            VoxelwiseValue.mni152_x == x,
            VoxelwiseValue.mni152_y == y,
            VoxelwiseValue.mni152_z == z
        )
    ).all()

    if not voxels:
        return []

    # Determine the correct table and file relationship based on map_type
    if map_type == 'connectivity':
        ArrayTable = ParcelwiseConnectivityValue
        FileTable = ConnectivityFile
        file_id_attr = 'connectivity_file_id'
    elif map_type == 'roi':
        ArrayTable = ParcelwiseROIValue
        FileTable = ROIFile
        file_id_attr = 'roi_file_id'
    elif map_type == 'group_level_map':
        ArrayTable = ParcelwiseGroupLevelMapValue
        FileTable = GroupLevelMapFile
        file_id_attr = 'group_level_map_files_id'
    else:
        raise ValueError(f"Unknown map_type: {map_type}. Valid types are 'connectivity', 'roi', 'group_level_map'.")

    results = []

    for voxel in voxels:
        # Query the appropriate ArrayTable for the given parcel_id
        arrays = session.query(ArrayTable).filter(ArrayTable.parcel_id == voxel.parcel_id).all()
        for array in arrays:
            # Query the FileTable for the given file ID
            file_id = getattr(array, file_id_attr)
            file = session.query(FileTable).filter(FileTable.id == file_id).first()
            if file:
                results.append((file.path, file.filetype, array.value))

    # Sort results by descending value
    results.sort(key=lambda x: x[2], reverse=True)

    return results

"""Functions for inserting data into SQL tables"""
def data_to_voxelwise_values_table(parcellation, session):
    """
    Converts a parcellation to a data array in MNI152 template space (using the fetch_2mm_mni152_mask mask).
    Needs columns for parcel_id, mni152_x, mni152_y, mni152_z, and user_id. The primary key is an incremented integer
    assigned by SQL.
    Use get_parcel_id to get the parcel_id, looking for the parcel with the same value and parcellation name.

        This is different from data_to_parcelwise_arrays_table because it is in MNI152 space, not in the parcellation space.

    Inserts into: `voxelwise_values` table in SQL.
    Saves a npz file with the voxel data.
    """
    mask_img = fetch_2mm_mni152_mask()
    masker = NiftiMasker(mask_img=mask_img).fit()
    
    parcellation_values = masker.transform(parcellation.maps).ravel().astype(int) # These are the parcel values at each voxel;
    mask_indices = np.nonzero(mask_img.get_fdata().astype(bool))
    coords = apply_affine(mask_img.affine, np.column_stack(mask_indices))
    parcel_id_map = {value: get_parcel_id(get_parcellation_id(parcellation.name, session), value, session) for value in np.unique(parcellation_values)}
    parcel_ids = np.array([parcel_id_map[value] for value in parcellation_values])
    results = np.column_stack((parcel_ids, coords))

    default_user_id = get_user_id(session)
    
    # Prepare a list of dictionaries for bulk insertion
    records = [
        {
            'parcel_id': row[0],
            'mni152_x': row[1],
            'mni152_y': row[2],
            'mni152_z': row[3],
            'user_id': default_user_id  # Set the default user_id
        }
        for row in results
    ]
    
    # Check for existing records and filter out the redundant ones
    existing_records = set(
        (v.parcel_id, v.mni152_x, v.mni152_y, v.mni152_z) for v in session.query(
            VoxelwiseValue.parcel_id, VoxelwiseValue.mni152_x, VoxelwiseValue.mni152_y, VoxelwiseValue.mni152_z
        ).all()
    )
    
    new_records = [record for record in records if (
        record['parcel_id'], record['mni152_x'], record['mni152_y'], record['mni152_z']) not in existing_records]
    
    if new_records:
        session.bulk_insert_mappings(VoxelwiseValue, new_records)
        session.commit()

def file_to_file_table(filepath, parcellation, map_type, session, 
                       statistic_type=None, 
                       control_cohort=None, 
                       threshold=None,
                       research_paper_id=None,
                       subject_id=None,
                       connectome_id=None,
                       is_surface=False,
                       is_volume=False,
                       is_2d=False,
                       is_3d=False,
                       override_existing=False):
    """
    Converts a file to a row in the connectivity_files, roi_files, or group_level_map_files table (depending on map_type).
    """
    # Determine the appropriate table and record structure based on map_type
    record = {'path': filepath, 'md5': md5_hash(filepath), 'user_id': get_user_id(session)}
    table, record_updates = None, {}

    if map_type == 'connectivity':
        table, record_updates = ConnectivityFile, {
            'subject_id': subject_id, 'connectome_id': connectome_id, 'is_surface': is_surface,
            'is_volume': is_volume, 'statistic_type': statistic_type
        }
    elif map_type == 'roi':
        table, record_updates = ROIFile, {
            'subject_id': subject_id, 'is_2d': is_2d, 'is_3d': is_3d
        }
    elif map_type == 'group_level_map':
        table, record_updates = GroupLevelMapFile, {
            'research_paper_id': research_paper_id, 'control_cohort': control_cohort, 'threshold': threshold
        }
    else:
        raise ValueError("map_type must be 'connectivity', 'roi', or 'group_level_map'.")

    record.update(record_updates)
    record['parcellation_id'] = get_parcellation_id(parcellation.name, session)
    record['filetype'] = determine_filetype(filepath)

    try:
        # Check for existing file by path or md5
        existing_file = session.query(table).filter((table.path == filepath) | (table.md5 == record['md5'])).first()

        if existing_file:
            if override_existing:
                # Delete existing file and associated arrays
                session.query(table).filter_by(id=existing_file.id).delete(synchronize_session=False)
                logger.info(f"Overriding and deleting existing file with path {filepath} or md5 {record['md5']}.")
            else:
                logger.info(f"File with path {filepath} or md5 {record['md5']} already exists. Not overriding.")
                return

        # Add the new record
        session.add(table(**record))
        session.commit()
        logger.info(f"File with path {filepath} added to the database.")

    except NoSuchTableError:
        logger.error(f"Table {table.__tablename__} does not exist in the database.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        session.rollback()

def data_to_parcelwise_arrays_table(voxelwise_map, parcellation, map_type, session, strategy='mean', voxelwise_map_name=None):
    """
    Converts a parcelwise data array from apply_parcellation to a dataframe.
    Inserts into: `parcelwise_connectivity_values`, `parcelwise_roi_values`, or `group_level_map_arrays` table in SQL.
    """
    try:
        if voxelwise_map_name is None:
            voxelwise_map_name = voxelwise_map
        # Apply parcellation and get parcel IDs
        parcelwise_array, region_ids = apply_parcellation(voxelwise_map, parcellation, strategy=strategy, return_region_ids=True)
        parcel_ids = np.array([get_parcel_id(get_parcellation_id(parcellation.name, session), value, session) for value in region_ids])
        parcel_ids_values = np.column_stack((parcel_ids, parcelwise_array.ravel()))

        # Determine the appropriate table and file table based on map_type
        if map_type == 'connectivity':
            table = ParcelwiseConnectivityValue
            file_table = ConnectivityFile
        elif map_type == 'roi':
            table = ParcelwiseROIValue
            file_table = ROIFile
        elif map_type == 'group_level_map':
            table = ParcelwiseGroupLevelMapValue
            file_table = GroupLevelMapFile
        else:
            raise ValueError("map_type must be 'connectivity', 'roi', or 'group_level_map'.")

        # Get file ID and default user ID
        file_id = get_file_id(voxelwise_map_name, file_table, session)
        default_user_id = get_user_id(session)

        # Prepare records for insertion
        records = [
            {
                f'{map_type}_file_id': file_id,
                'parcel_id': row[0],
                'value': row[1],
                'user_id': default_user_id  # Set the default user_id
            }
            for row in parcel_ids_values
        ]

        # Check for existing records
        existing_records = set(
            (r.parcel_id, r.value) for r in session.query(
                table.parcel_id, table.value
            ).filter_by(**{f'{map_type}_file_id': file_id}).all()
        )

        # Filter new records to insert
        new_records = [record for record in records if (record['parcel_id'], record['value']) not in existing_records]

        # Insert new records if any
        if new_records:
            session.bulk_insert_mappings(table, new_records)
            session.commit()
            print(f"Data added to the {map_type}_arrays table in the database.")
        else:
            print("No new records to add")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()
    
def parcellation_to_parcellation_table(parcellation, session: _Session):
    """
    Converts a parcellation to a row in the parcellations table.
    Parcellation: a nilearn parcellation object. Must have name, description, path.
    """
    try:
        # Check if the parcellation already exists
        existing_parcellation = session.query(Parcellation).filter_by(name=parcellation.name).first()
        
        if existing_parcellation:
            print(f"Parcellation with name {parcellation.name} already exists.")
            return
        
        default_user_id = get_user_id(session)
        
        record = {
            'name': parcellation.name,
            'description': parcellation.description,
            'user_id': default_user_id  # Set the default user_id
        }
        
        if getattr(parcellation, 'filename', None):
            record['path'] = parcellation.filename
            record['md5'] = md5_hash(parcellation.filename)
        else:
            record['md5'] = md5_hash(parcellation.maps)
        
        session.add(Parcellation(**record))
        session.commit()
        print(f"Parcellation with name {parcellation.name} added to the database.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def delete_dependent_arrays_and_return_data(parcellation_id: int, session: _Session) -> dict:
    data = {
        'voxelwise_values': [],
        'parcelwise_connectivity_values': [],
        'parcelwise_roi_values': [],
        'group_level_map_arrays': []
    }
    
    try:
        # Delete and return data for VoxelwiseValue
        voxelwise_values = session.query(VoxelwiseValue).join(Parcel).filter(Parcel.parcellation_id == parcellation_id).all()
        for v in voxelwise_values:
            data['voxelwise_values'].append({
                'id': v.id,
                'mni152_x': v.mni152_x,
                'mni152_y': v.mni152_y,
                'mni152_z': v.mni152_z,
                'parcel_id': v.parcel_id,
                'user_id': v.user_id
            })
        session.query(VoxelwiseValue).filter(VoxelwiseValue.id.in_([v.id for v in voxelwise_values])).delete(synchronize_session=False)

        # Delete and return data for ParcelwiseConnectivityValue
        parcelwise_connectivity_values = session.query(ParcelwiseConnectivityValue).join(Parcel).filter(Parcel.parcellation_id == parcellation_id).all()
        for c in parcelwise_connectivity_values:
            data['parcelwise_connectivity_values'].append({
                'id': c.id,
                'value': c.value,
                'connectivity_file_id': c.connectivity_file_id,
                'parcel_id': c.parcel_id,
                'user_id': c.user_id
            })
        session.query(ParcelwiseConnectivityValue).filter(ParcelwiseConnectivityValue.id.in_([c.id for c in parcelwise_connectivity_values])).delete(synchronize_session=False)

        # Delete and return data for ParcelwiseROIValue
        parcelwise_roi_values = session.query(ParcelwiseROIValue).join(Parcel).filter(Parcel.parcellation_id == parcellation_id).all()
        for r in parcelwise_roi_values:
            data['parcelwise_roi_values'].append({
                'id': r.id,
                'value': r.value,
                'roi_file_id': r.roi_file_id,
                'parcel_id': r.parcel_id,
                'user_id': r.user_id
            })
        session.query(ParcelwiseROIValue).filter(ParcelwiseROIValue.id.in_([r.id for r in parcelwise_roi_values])).delete(synchronize_session=False)

        # Delete and return data for ParcelwiseGroupLevelMapValue
        group_level_map_arrays = session.query(ParcelwiseGroupLevelMapValue).join(Parcel).filter(Parcel.parcellation_id == parcellation_id).all()
        for g in group_level_map_arrays:
            data['group_level_map_arrays'].append({
                'id': g.id,
                'value': g.value,
                'group_level_map_files_id': g.group_level_map_files_id,
                'parcel_id': g.parcel_id,
                'user_id': g.user_id
            })
        session.query(ParcelwiseGroupLevelMapValue).filter(ParcelwiseGroupLevelMapValue.id.in_([g.id for g in group_level_map_arrays])).delete(synchronize_session=False)
        
        session.commit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()
    
    return data

def reinsert_dependent_arrays(data: dict, session: _Session):
    """
    Reinserts dependent arrays into their respective tables.
    """
    try:
        if data['voxelwise_values']:
            session.bulk_insert_mappings(VoxelwiseValue, data['voxelwise_values'])
        
        if data['parcelwise_connectivity_values']:
            session.bulk_insert_mappings(ParcelwiseConnectivityValue, data['parcelwise_connectivity_values'])
        
        if data['parcelwise_roi_values']:
            session.bulk_insert_mappings(ParcelwiseROIValue, data['parcelwise_roi_values'])
        
        if data['group_level_map_arrays']:
            session.bulk_insert_mappings(ParcelwiseGroupLevelMapValue, data['group_level_map_arrays'])
        
        session.commit()
        print("Finished reinserting dependent arrays.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def parcellation_to_parcels_table(parcellation: Bunch, session: _Session, override_existing: Optional[bool] = False):
    """
    Converts a parcellation to a list of parcels, and inserts them into the parcels table.
    The parcels table has columns id, value, label, parcellation_id, along with user_id and insert_date.
    The function works just fine, but we want to add the option for override_existing. 
    This could be complex because voxelwise_values, parcelwise_connectivity_values, parcelwise_roi_values, and group_level_map_arrays all depend on parcels.
    We want a function that deletes the arrays that depend on parcels, but returns the data in them as a dictionary, so that
    we can re-build them after the parcels are re-inserted (overridden).
    """
    try:
        # Ensure the parcellation record exists and get its ID
        parcellation_record = session.query(Parcellation).filter_by(name=parcellation.name).first()
        if not parcellation_record:
            parcellation_to_parcellation_table(parcellation, session)
            parcellation_record = session.query(Parcellation).filter_by(name=parcellation.name).first()
        
        parcellation_id = parcellation_record.id

        # Prepare the masker and labels
        image = fetch_2mm_mni152_mask()
        masker = NiftiLabelsMasker(labels_img=parcellation.maps, labels=parcellation.labels, strategy='mean')
        masker.fit_transform(image)
        labels = parcellation.labels
        parcel_values = masker.labels_img_.get_fdata().ravel()
        unique_values = np.unique(parcel_values)
        
        if len(labels) != len(unique_values):
            warnings.warn("Mismatch between the number of labels and unique parcel values. "
                          "Got {} labels and {} unique values.".format(len(labels), len(unique_values)))
        
        parcel_value_label_pairs = {int(value): labels[int(value)] for value in unique_values}

        # Get default user ID
        default_user_id = get_user_id(session)

        # Build the parcel records
        all_records = [
            {'parcellation_id': parcellation_id, 'value': int(value), 'label': label, 'user_id': default_user_id}
            for value, label in parcel_value_label_pairs.items()
        ]

        if override_existing:
            dependent_data = delete_dependent_arrays_and_return_data(parcellation_id, session)
            session.query(Parcel).filter_by(parcellation_id=parcellation_id).delete(synchronize_session=False)
            session.bulk_insert_mappings(Parcel, all_records)
            reinsert_dependent_arrays(dependent_data, session)
        else:
            existing_parcels = set(
                (p.parcellation_id, p.value) for p in session.query(Parcel.parcellation_id, Parcel.value)
                .filter_by(parcellation_id=parcellation_id).all()
            )

            new_records = [
                record for record in all_records if (record['parcellation_id'], record['value']) not in existing_parcels
            ]

            if new_records:
                session.bulk_insert_mappings(Parcel, new_records)
        
        session.commit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

"""
Functions to pre-populate certain tables with data from JSON files.
"""

def remove_symptom_from_db(id: int, session: _Session) -> dict:
    """
    Removes a symptom and associated synonyms from the database, but returns a JSON copy of the symptom data and its synonyms.

    Args:
        id (int): The ID of the symptom to remove.
        session (Session): The SQLAlchemy session object for database operations.

    Returns:
        dict: A dictionary containing the symptom data and its synonyms.
    """
    # Query the symptom by ID
    symptom = session.query(Symptom).filter_by(id=id).one_or_none()
    
    if not symptom:
        raise ValueError(f"No symptom found with ID {id}")

    # Extract symptom data
    symptom_data = {
        "name": symptom.name,
        "description": symptom.description,
        "domain_id": symptom.domain_id,
        "subdomain_id": symptom.subdomain_id,
        "synonyms": [synonym.name for synonym in symptom.synonyms],
        "mesh_terms": [mesh_term.name for mesh_term in symptom.mesh_terms]
    }

    # Remove associated synonyms and mesh terms
    session.query(Synonym).filter_by(symptom_id=id).delete()
    session.query(MeshTerm).filter_by(symptom_id=id).delete()

    # Remove the symptom
    session.delete(symptom)
    session.commit()

    # Update user_id for tracking
    default_user_id = get_user_id(session)
    symptom.user_id = default_user_id
    session.commit()

    return symptom_data

def insert_dimensions_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts statistic types into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing dimension data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    
    Returns:
        None
    """
    try:
        with open(json_path, 'r') as file:
            dimensions_data = json.load(file)

        default_user_id = get_user_id(session)

        for dimension in dimensions_data:
            existing_dimension = session.execute(
                select(Dimension).filter_by(name=dimension['name'])
            ).scalar_one_or_none()

            if existing_dimension is None:
                new_dimension = Dimension(
                    name=dimension.get('name'),
                    description=dimension.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_dimension)
            elif override_existing:
                session.delete(existing_dimension)
                session.flush()
                new_dimension = Dimension(
                    name=dimension.get('name'),
                    description=dimension.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_dimension)

        session.commit()

        dimensions = session.query(Dimension).all()
        print(f"There are now {len(dimensions)} dimensions in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()   

def insert_modalities_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts modalities into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing modality data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    
    Returns:
        None
    """
    try:
        with open(json_path, 'r') as file:
            modalities_data = json.load(file)

        default_user_id = get_user_id(session)

        for modality in modalities_data:
            existing_modality = session.execute(
                select(ImageModality).filter_by(name=modality['name'])
            ).scalar_one_or_none()

            if existing_modality is None:
                new_modality = ImageModality(
                    name=modality.get('name'),
                    description=modality.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_modality)
            elif override_existing:
                session.delete(existing_modality)
                session.flush()
                new_modality = ImageModality(
                    name=modality.get('name'),
                    description=modality.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_modality)

        session.commit()

        modalities = session.query(ImageModality).all()
        print(f"There are now {len(modalities)} modalities in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def insert_statistic_types_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts statistic types into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing statistic type data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    
    Returns:
        None
    """
    try:
        with open(json_path, 'r') as file:
            statistic_types_data = json.load(file)

        default_user_id = get_user_id(session)

        for statistic_type in statistic_types_data:
            existing_statistic_type = session.execute(
                select(StatisticType).filter_by(name=statistic_type['name'])
            ).scalar_one_or_none()

            if existing_statistic_type is None:
                new_statistic_type = StatisticType(
                    name=statistic_type.get('name'),
                    code=statistic_type.get('code'),
                    description=statistic_type.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_statistic_type)
            elif override_existing:
                session.delete(existing_statistic_type)
                session.flush()
                new_statistic_type = StatisticType(
                    name=statistic_type.get('name'),
                    code=statistic_type.get('code'),
                    description=statistic_type.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_statistic_type)

        session.commit()

        statistic_types = session.query(StatisticType).all()
        print(f"There are now {len(statistic_types)} statistic types in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()            

def insert_connectomes_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts connectomes into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing connectome data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_path, 'r') as file:
            connectomes_data = json.load(file)

        default_user_id = get_user_id(session)

        for connectome in connectomes_data:
            existing_connectome = session.execute(
                select(Connectome).filter_by(name=connectome['name'])
            ).scalar_one_or_none()

            if existing_connectome is None:
                new_connectome = Connectome(
                    name=connectome.get('name'),
                    connectome_type=connectome.get('connectome_type'),
                    description=connectome.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_connectome)
            elif override_existing:
                session.delete(existing_connectome)
                session.flush()  # Ensure the existing connectome is removed
                new_connectome = Connectome(
                    name=connectome.get('name'),
                    connectome_type=connectome.get('connectome_type'),
                    description=connectome.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_connectome)

        session.commit()

        connectomes = session.query(Connectome).all()
        print(f"There are now {len(connectomes)} connectomes in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def insert_handedness_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts handedness into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing handedness data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_path, 'r') as file:
            handedness_data = json.load(file)

        default_user_id = get_user_id(session)

        for handedness in handedness_data:
            existing_handedness = session.execute(
                select(Handedness).filter_by(name=handedness['name'])
            ).scalar_one_or_none()

            if existing_handedness is None:
                new_handedness = Handedness(
                    name=handedness.get('name'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_handedness)
            elif override_existing:
                session.delete(existing_handedness)
                session.flush()
                new_handedness = Handedness(
                    name=handedness.get('name'),
                    user_id=default_user_id
                )
                session.add(new_handedness)

        session.commit()

        handedness_records = session.query(Handedness).all()
        print(f"There are now {len(handedness_records)} handedness records in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def insert_sexes_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts sexes into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing sex data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_path, 'r') as file:
            sexes_data = json.load(file)

        default_user_id = get_user_id(session)

        for sex in sexes_data:
            existing_sex = session.execute(
                select(Sex).filter_by(name=sex['name'])
            ).scalar_one_or_none()

            if existing_sex is None:
                new_sex = Sex(
                    name=sex.get('name'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_sex)
            elif override_existing:
                session.delete(existing_sex)
                session.flush()  # Ensure the existing sex is removed
                new_sex = Sex(
                    name=sex.get('name'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_sex)

        session.commit()

        sexes = session.query(Sex).all()
        print(f"There are now {len(sexes)} sexes in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def insert_domains_from_json(json_file_path: str, db_session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts domains and subdomains into the database, avoiding duplicates.

    Args:
        json_file_path (str): The path to the JSON file containing domain data.
        db_session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        default_user_id = get_user_id(db_session)

        for domain_data in data:
            existing_domain = db_session.execute(
                select(Domain).filter_by(name=domain_data['name'])
            ).scalar_one_or_none()

            if existing_domain and override_existing:
                # Remove all symptoms associated with the existing domain
                associated_symptoms = db_session.query(Symptom).filter_by(domain_id=existing_domain.id).all()
                for symptom in associated_symptoms:
                    remove_symptom_from_db(symptom.id, db_session)
                
                # Delete all subdomains associated with the existing domain
                db_session.query(Subdomain).filter_by(domain_id=existing_domain.id).delete()
                db_session.flush()  # Ensure subdomains are removed

                # Delete the existing domain
                db_session.delete(existing_domain)
                db_session.flush()  # Ensure the existing domain is removed
                existing_domain = None  # Clear the reference to the existing domain

            if existing_domain is None:
                domain = Domain(
                    name=domain_data['name'],
                    description=domain_data.get('description', None),
                    user_id=default_user_id  # Set the default user_id
                )
                db_session.add(domain)
                db_session.flush()  # Ensure domain.id is populated
            else:
                domain = existing_domain

            for subdomain_data in domain_data.get('subdomains', []):
                existing_subdomain = db_session.execute(
                    select(Subdomain).filter_by(name=subdomain_data['name'], domain_id=domain.id)
                ).scalar_one_or_none()

                if existing_subdomain and override_existing:
                    db_session.delete(existing_subdomain)
                    db_session.flush()  # Ensure the existing subdomain is removed
                    existing_subdomain = None  # Clear the reference to the existing subdomain

                if existing_subdomain is None:
                    subdomain = Subdomain(
                        name=subdomain_data['name'],
                        description=subdomain_data.get('description', None),
                        domain_id=domain.id,
                        user_id=default_user_id  # Set the default user_id for subdomains
                    )
                    db_session.add(subdomain)

        db_session.commit()

        domains = db_session.query(Domain).all()
        print(f"There are now {len(domains)} domains in the database.")

        subdomains = db_session.query(Subdomain).all()
        print(f"There are now {len(subdomains)} subdomains in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        db_session.rollback()
    finally:
        db_session.close()

def insert_causes_from_json(json_path: str, session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts causes into the database, avoiding duplicates.

    Args:
        json_path (str): The path to the JSON file containing cause data.
        session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_path, 'r') as file:
            causes_data = json.load(file)

        default_user_id = get_user_id(session)

        for cause in causes_data:
            existing_cause = session.execute(
                select(Cause).filter_by(name=cause['name'])
            ).scalar_one_or_none()

            if existing_cause is None:
                new_cause = Cause(
                    name=cause.get('name'),
                    description=cause.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_cause)
            elif override_existing:
                session.delete(existing_cause)
                session.flush()  # Ensure the existing cause is removed
                new_cause = Cause(
                    name=cause.get('name'),
                    description=cause.get('description'),
                    user_id=default_user_id  # Set the default user_id
                )
                session.add(new_cause)

        session.commit()

        causes = session.query(Cause).all()
        print(f"There are now {len(causes)} causes in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()

def insert_symptoms_from_json(json_file_path: str, db_session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts symptoms into the database, avoiding duplicates.

    Args:
        json_file_path (str): The path to the JSON file containing symptom data.
        db_session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        default_user_id = get_user_id(db_session)

        for symptom_data in data:
            # Find the corresponding Domain ID
            domain = db_session.execute(
                select(Domain).filter_by(name=symptom_data['domain'])
            ).scalar_one_or_none()

            if not domain:
                print(f"Domain '{symptom_data['domain']}' not found for symptom '{symptom_data['name']}'")
                continue

            # Find the corresponding Subdomain ID
            subdomain = db_session.execute(
                select(Subdomain).filter_by(name=symptom_data['subdomain'], domain_id=domain.id)
            ).scalar_one_or_none()

            if not subdomain:
                print(f"Subdomain '{symptom_data['subdomain']}' not found for symptom '{symptom_data['name']}'")
                continue

            # Check if the symptom already exists
            existing_symptom = db_session.execute(
                select(Symptom).filter_by(name=symptom_data['name'], domain_id=domain.id, subdomain_id=subdomain.id)
            ).scalar_one_or_none()

            if existing_symptom and override_existing:
                # Delete existing synonyms and mesh terms first
                db_session.query(Synonym).filter_by(symptom_id=existing_symptom.id).delete()
                db_session.query(MeshTerm).filter_by(symptom_id=existing_symptom.id).delete()
                db_session.flush()
                # Delete the existing symptom
                db_session.delete(existing_symptom)
                db_session.flush()
                existing_symptom = None

            if existing_symptom is None:
                # Insert the new symptom
                symptom = Symptom(
                    name=symptom_data['name'],
                    description=symptom_data.get('description', None),
                    domain_id=domain.id,
                    subdomain_id=subdomain.id,
                    user_id=default_user_id  # Set the default user_id
                )
                db_session.add(symptom)
                db_session.flush()  # Ensure symptom.id is populated

                # Insert synonyms
                for synonym_name in symptom_data.get('synonyms', []):
                    synonym = Synonym(
                        name=synonym_name,
                        symptom_id=symptom.id,
                        user_id=default_user_id  # Set the default user_id
                    )
                    db_session.add(synonym)

                # Insert mesh terms
                for mesh_term_name in symptom_data.get('mesh_terms', []):
                    mesh_term = MeshTerm(
                        name=mesh_term_name,
                        symptom_id=symptom.id,
                        user_id=default_user_id  # Set the default user_id
                    )
                    db_session.add(mesh_term)

        db_session.commit()

        symptoms = db_session.query(Symptom).all()
        print(f"There are now {len(symptoms)} symptoms in the database.")

        synonyms = db_session.query(Synonym).all()
        print(f"There are now {len(synonyms)} synonyms in the database.")

        mesh_terms = db_session.query(MeshTerm).all()
        print(f"There are now {len(mesh_terms)} mesh terms in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        db_session.rollback()
    finally:
        db_session.close()

def insert_cohorts_from_json(json_file_path: str, db_session: _Session, override_existing: Optional[bool] = False):
    """
    Parses a JSON file and inserts cohorts into the database, avoiding duplicates.

    Args:
        json_file_path (str): The path to the JSON file containing cohort data.
        db_session (Session): The SQLAlchemy session object for database operations.
        override_existing (bool): If true, existing records will be replaced with new data.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        for cohort_data in data:
            # Check if the cohort already exists
            existing_cohort = db_session.execute(
                select(PatientCohort).filter_by(name=cohort_data['name'])
            ).scalar_one_or_none()

            if existing_cohort and override_existing:
                db_session.delete(existing_cohort)
                db_session.flush()
                existing_cohort = None

            if existing_cohort is None:
                # Insert the new cohort
                cohort = PatientCohort(
                    user_id=get_user_id(db_session),
                    name=cohort_data['name'],
                    description=cohort_data.get('description', None)
                )
                db_session.add(cohort)

        db_session.commit()

        cohorts = db_session.query(PatientCohort).all()
        print(f"There are now {len(cohorts)} cohorts in the database.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        db_session.rollback()
    finally:
        db_session.close()

def process_parcellation(parcellation: Bunch, session: _Session):
    """
    Process a parcellation and insert it into the database.
    """
    parcellation_to_parcellation_table(parcellation, session)
    parcellation_to_parcels_table(parcellation, session)
    data_to_voxelwise_values_table(parcellation, session)


"""Functions used to build nimlabd database (Authored August 22nd, 2024 by Joseph Turner)"""

required_entities_with_defaults = {
    'datatype':'unknown',
    'connectome':None,
    'statistic':None,
    'coordinate_space':'unknown',
    'hemisphere':'unknown',
    'roi_size':None,
    'roi_type':'unknown',
    'image_type':'unknown',
    'roi_dimensionality':'unknown',
    'md5':None, 
    'extension':'unknown',
    'mask':'unknown',
}

def get_or_create_subject(session: _Session, subject_nickname: str, research_paper: ResearchPaper, age: Optional[int], case_report: Optional[CaseReport], patient_cohort: Optional[PatientCohort], sex: Optional[Sex], cause: Optional[Cause], handedness: Optional[Handedness]) -> Subject:
    """Creates a new subject, or finds an existing subject with the same nickname and belonging either to the same research paper, patient cohort, or case report."""

    # Filter out None values to avoid querying with None as a potential issue
    filters = [
        Subject.nickname == subject_nickname,
        or_(
            ResearchPaper.id == research_paper.id,
            Subject.patient_cohort == patient_cohort if patient_cohort else False,
            Subject.case_report == case_report if case_report else False
        )
    ]
    
    subject = session.query(Subject).join(ResearchPaper.subjects).filter(*filters).first()

    if not subject:
        subject = Subject(
            user_id=get_user_id(session),
            nickname=subject_nickname,
            age=age,
            case_report=case_report,
            patient_cohort=patient_cohort,
            sex=get_or_create_sex(session, name=sex) if sex else None,
            handedness=get_or_create_handedness(session, name=handedness) if handedness else None,
            cause=get_or_create_cause(session, name=cause) if cause else None
        )
        session.add(subject)
        session.flush()  # Flush immediately to ensure it's added to the session

    return subject

def get_or_create_research_paper(session: _Session, title: str, nickname: str, first_author: Author,  doi: Optional[str], comments: Optional[str]) -> ResearchPaper:
    """Creates a new research paper, or returns an existing one if it already exists."""
    research_paper = session.query(ResearchPaper).filter(
        or_(
            ResearchPaper.doi == doi,
            ResearchPaper.nickname == nickname,
            ResearchPaper.title == title
        )
    ).first()

    if not research_paper:
        user_id = get_user_id(session)
        research_paper = ResearchPaper(
            doi=doi,
            title=title,
            nickname=nickname,
            user_id=user_id,
            first_author=first_author,
            comments=comments
        )
        session.add(research_paper)
        session.flush()

    return research_paper

def get_or_create_case_report(session: _Session, doi:Optional[str]=None, pubmed_id:Optional[int]=None, other_citation:Optional[str]=None, path:Optional[str]=None, title:Optional[str]=None) -> CaseReport:
    """Creates a new case report, or returns an existing one if it already exists."""
    if all(v is None for v in [doi, pubmed_id, other_citation, path, title]):
        print(f"No identifying information found for case report {doi, pubmed_id, other_citation, path, title}. Skipping.")
        return None
    case_report = session.query(CaseReport).filter_by(
        doi = doi,
        pubmed_id = pubmed_id,
        other_citation = other_citation,
        path = path,
        title = title
    ).first()
    if not case_report:
        case_report = CaseReport(
            doi=doi,
            pubmed_id=pubmed_id,
            other_citation=other_citation,
            path=path,
            title=title,
            user_id=get_user_id(session)
        )
        session.add(case_report)
        session.flush()
    return case_report

def get_or_create_cause(session: _Session, name: str) -> Cause:
    """Checks if a Cause with the given name exists, and creates a new one if it doesn't."""
    if name is None:
        name = "unspecified"
    try:
        cause = session.query(Cause).filter(Cause.name == name).one()
    except NoResultFound:
        cause = Cause(name=name, user_id=get_user_id(session))
        session.add(cause)
        session.flush()
        print(f"Created new cause: {name}")
    return cause

def get_or_create_handedness(session: _Session, name: str) -> Handedness:
    """Checks if a Handedness with the given name exists, and creates a new one if it doesn't."""
    if name is None:
        name = "unknown"
    try:
        handedness = session.query(Handedness).filter(Handedness.name == name).one()
    except NoResultFound:
        handedness = Handedness(name=name, user_id=get_user_id(session))
        session.add(handedness)
        session.flush()
        print(f"Created new handedness: {name}")
    return handedness

def get_or_create_sex(session: _Session, name: str) -> Sex:
    """Checks if a sex with the given name exists, and creates a new one if it doesn't."""
    if name is None:
        name = "unknown"
    try: 
        sex = session.query(Sex).filter(Sex.name == name).one()
    except NoResultFound:
        sex = Sex(name=name, user_id=get_user_id(session))
        session.add(sex)
        session.flush()
        print(f"Created new sex: {name}")
    return sex

def get_or_create_coordinate_space(session: _Session, name: str) -> CoordinateSpace:
    """Checks if a CoordinateSpace with the given name exists, and creates a new one if it doesn't."""
    if name is None:
        name = "unknown"
    try:
        coordinate_space = session.query(CoordinateSpace).filter(CoordinateSpace.name == name).one()
    except NoResultFound:
        coordinate_space = CoordinateSpace(name=name, user_id=get_user_id(session))
        session.add(coordinate_space)
        session.flush()
        print(f"Created new coordinate space: {name}")
    return coordinate_space

def get_or_create_author(session: _Session, name: str, email: Optional[str]) -> Author:
    """Checks if an Author with the given name exists, and creates a new one if it doesn't."""
    try:
        author = session.query(Author).filter(Author.name == name).one()
    except NoResultFound:
        author = Author(name=name, email=email, user_id=get_user_id(session))
        session.add(author)
        session.flush()  # This will assign an ID to the new author
        print(f"Created new author: {name}")
    return author

def get_or_create_connectome(session: _Session, name: str, description: Optional[str] = None, override_existing: bool = False) -> Connectome:
    """Checks if a Connectome with the given name exists, and creates a new one if it doesn't."""
    if description is None:
        description = "No description provided, as this is was a connectome instance automatically parsed from the data without a description."
    
    connectome = session.query(Connectome).filter(func.lower(Connectome.name) == name.lower()).first()
    if connectome:
        if override_existing:
            connectome.description = description
            session.commit()
            print(f"Updated existing connectome: {name}")
        return connectome
    else:
        connectome = Connectome(
            name=name, 
            description=description, 
            user_id=get_user_id(session))
        session.add(connectome)
        session.flush()
        print(f"Created new connectome: {name}")
        return connectome

def get_or_create_statistic_type(session: _Session, code: str, name: Optional[str] = None, description: Optional[str] = None, override_existing: bool = False) -> StatisticType:
    """
    Checks if a StatisticType with the given code exists, and creates a new one if it doesn't.
    If override_existing is True, it will update the name and description of the existing StatisticType.
    """
    # Use the provided code to search for an existing StatisticType
    statistic_type = session.query(StatisticType).filter(StatisticType.code == code).first()

    if statistic_type:
        # If the StatisticType exists and override_existing is True, update the name and description
        if override_existing:
            if name is not None:
                statistic_type.name = name
            if description is not None:
                statistic_type.description = description
            session.commit()
            print(f"Updated existing StatisticType with code: {code}")
    else:
        # If the StatisticType doesn't exist, create a new one
        statistic_type = StatisticType(
            code=code,
            name=name or f"StatisticType for code: {code}",
            description=description or "No description provided.",
            user_id=get_user_id(session)
        )
        session.add(statistic_type)
        session.flush()  # This will assign an ID to the new StatisticType
        print(f"Created new StatisticType with code: {code}")

    return statistic_type

def process_research_paper_to_db(session, name, nickname, comments, doi, first_author, authors_contacts_dict):
    # Process first author first
    first_author_name, first_author_email = list(first_author.items())[0]
    first_author = get_or_create_author(session=session, name=first_author_name, email=first_author_email)
    
    # Get or create the research paper
    research_paper = get_or_create_research_paper(session=session, title=name, nickname=nickname, first_author=first_author, doi=doi, comments=comments)

    if first_author not in research_paper.authors:
        research_paper.authors.append(first_author)

    # Process all authors, including the first author
    for author_name, author_email in authors_contacts_dict.items():
        author_obj = get_or_create_author(session=session, name=author_name, email=author_email)
        if author_obj not in research_paper.authors:
            research_paper.authors.append(author_obj)

    # Commit the changes
    try:
        session.commit()
        print(f"Successfully processed research paper: {nickname}")
    except Exception as e:
        session.rollback()
        print(f"Error processing research paper: {e}")
        raise

    return research_paper

def associate_authors_with_contacts(authors, contacts):
    associations = {}
    for author in authors:
        match, score, _ = process.extractOne(author, contacts, scorer=fuzz.partial_ratio)
        if score > 50:
            associations[author] = match
        else:
            associations[author] = None
    return associations

def determine_first_author(name, authors):
    match, score, _ = process.extractOne(name, authors, scorer=fuzz.partial_ratio)
    return match

def get_subdirs(path):
    valid_subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')]
    valid_subdirs = [d for d in valid_subdirs if 'input' not in d]
    return valid_subdirs

def get_derivatives(dir):
    if 'derivatives/nimlab-preprocessing' not in dir:
        return get_derivatives(os.path.join(dir, 'derivatives/nimlab-preprocessing'))
    
    if any('sub-' in s for s in os.listdir(dir)):
        return [dir]
    
    return [
        subdir for subdir in get_subdirs(dir)
        if any('sub-' in f for f in os.listdir(os.path.join(dir, subdir)))
    ]

def process_cohort(session, cohort_name):
    """
    Determines if a cohort_name is a valid cohort in the database. 
    Params:
        cohort_name: str
            The name of the cohort to check.
    Returns:
        bool: 
            True if the cohort_name is a valid cohort in the database, False otherwise.
        cohort_name: str
            The name of the cohort if it is a valid cohort in the database, otherwise returns the input cohort_name.
    """    
    with open('/data/nimlab/USERS/joseph/nimlab_db_development/db_utils/data/cohorts.json', 'r') as file:
        cohort_data = json.load(file)

    name_to_cohort = {cohort['name']: cohort['name'] for cohort in cohort_data}
    for cohort in cohort_data:
        synonyms = cohort['synonyms']
        if isinstance(synonyms, list):
            name_to_cohort.update({syn: cohort['name'] for syn in synonyms})
        else:
            name_to_cohort[synonyms] = cohort['name']
    
    best_match, score, _ = process.extractOne(cohort_name, list(name_to_cohort.keys()), scorer=fuzz.ratio)
    
    if score >= 70:
        match = name_to_cohort[best_match]
        cohort = session.query(PatientCohort).filter_by(name=match).first()
        return True, cohort
    else:
        return False, cohort_name
        
def process_research_paper(session, dataset_description_paths):
    name = dataset_description_paths[0].replace('/data/nimlab/NIMLAB_DATABASE/published_datasets_v2/', '').split('/')[0]

    authors, nicknames, dois, tags, contacts, comments = [], [], [], [], [], []

    for dataset_description_path in dataset_description_paths:
        with open(dataset_description_path, 'r') as file:
            data = json.load(file)

        authors.extend(data.get('authors', []))
        nicknames.append(data.get('Name', ''))
        contacts.extend(data.get('contacts', []))
        comments.append(data.get('comments', ''))
        dois.extend(data.get('DOI', []))
        tags.extend(data.get('tags', []))

    # Remove duplicates by converting to set and back to list
    authors = list(set(authors))
    nicknames = list(set(nicknames))
    contacts = list(set(contacts))
    comments = list(set(comments))
    dois = list(set(dois))
    tags = list(set(tags))

    # Remove any empty strings
    nicknames = [n for n in nicknames if n]
    comments = [c for c in comments if c]

    first_author = determine_first_author(name, authors)
    authors_contacts_dict = associate_authors_with_contacts(authors, contacts)
    first_author_contact = authors_contacts_dict.get(first_author, None)
    first_author = {first_author: first_author_contact}

    nickname = ', '.join(nicknames)
    comments = ' AND '.join(comments)
    doi = dois[0] if dois else None
    if len(dois) > 1:
        print(f"WARNING: Found multiple DOIs: {dois}")
        
    # Process the research paper
    research_paper = process_research_paper_to_db(session, name, nickname, comments, doi, first_author, authors_contacts_dict)
    return research_paper

def process_literature_files(path):
    # print(f"Processing literature at path: {path}")
    case_reports_paths = []
    research_paper_paths = []
    if os.path.exists(os.path.join(path, 'caseReports')):
        case_reports_paths = [os.path.join(path, 'caseReports', case_report) for case_report in os.listdir(os.path.join(path, 'caseReports'))]
    if os.path.exists(os.path.join(path, 'paper')):
        research_paper_paths = [os.path.join(path, 'paper', paper) for paper in os.listdir(os.path.join(path, 'paper'))]
    return case_reports_paths, research_paper_paths

def process_subject_json(session, filepath, literature_dict, research_paper, cohort_obj=None):
    """Docstring to be developed soon."""
    def load_and_process_json(filepath):
        """Loads the subject JSON and removes the 'id' key if it exists."""
        with open(filepath, 'r') as file:
            data = json.load(file)
        data.pop('id', None)
        return data

    def get_subject_name(filepath):
        """Extracts the subject name from the filepath."""
        subject_name = os.path.basename(filepath).replace('.json', '').replace('sub-', '')
        if not subject_name:
            raise ValueError(f"No subject identifier found in {filepath}")
        return subject_name

    def process_literature_data(data, literature_dict, filepath):
        """If present, finds doi/pmid/other citation information for this subject. If provided, matches this information to the literature files in the sourcedata."""
        doi = data.pop('DOI', None)
        if doi:
            doi = doi.strip()  # Remove leading/trailing whitespace
            # Case-insensitive replace and remove common prefixes
            doi = doi.replace("DOI: ", "", 1).replace("doi: ", "", 1).replace("https://doi.org/", "", 1)
            data['doi'] = doi.strip()  # Ensure no residual spaces
            data['doi'] = data['doi'] if data['doi'].isdigit() else data['doi']
        else:
            data['doi'] = None

        data['pubmed_id'] = data.pop('PMID', None)
        data['pubmed_id'] = (
            int(str(data['pubmed_id']).split('.')[0])
            if data['pubmed_id'] 
            else None
        )
        data['other_citation'] = data.pop('citation', None)
        data['path'] = find_paper_path(data.get('paper'), literature_dict, filepath)
        data['title'] = data.get('paper').replace('.pdf', '').replace('.docx', '').replace('.doc', '') if data.get('paper') else None
        data.pop('paper', None)
        return data

    def find_paper_path(paper_name, literature_dict, filepath):
        """Finds the full path to the research paper for this subject."""
        if literature_dict.get('case_reports') and paper_name:
            case_report_files = literature_dict['case_reports']
            if paper_name in [os.path.basename(file) for file in case_report_files]:
                return os.path.join(os.path.dirname(filepath), 'sourcedata/literature/caseReports', paper_name)
        return None

    def associate_subject_with_research_paper(session, subject, research_paper):
        if subject not in research_paper.subjects:
            research_paper.subjects.append(subject)

    def commit_changes(session):
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            raise e

    # Main process flow
    data = load_and_process_json(filepath)
    subject_name = get_subject_name(filepath)
    
    data.update(process_literature_data(data, literature_dict, filepath))
    data['user_id'] = get_user_id(session)
    
    case_report = get_or_create_case_report(session=session,
                                            doi=data.get('doi'), 
                                            pubmed_id=data.get('pubmed_id'), 
                                            other_citation=data.get('other_citation'), 
                                            path=data.get('path'), 
                                            title=data.get('title'))
    subject = get_or_create_subject(session=session, 
                                    subject_nickname=subject_name, 
                                    case_report=case_report, 
                                    patient_cohort=cohort_obj, 
                                    research_paper=research_paper, 
                                    sex=data.get('sex', None),
                                    cause=data.get('cause', None),
                                    handedness=data.get('handedness', None),
                                    age=data.get('age', None)
    )
    
    associate_subject_with_research_paper(session, subject, research_paper)
    commit_changes(session)

    print(f"Processed subject {subject_name} from paper {research_paper.nickname}")
    return subject

def process_imaging_file(session, filepath, entities, metadata, subject_obj):
    """
    Fetch generic metadata for a neuroimaging file.
    """
    extension = entities.get('extension', required_entities_with_defaults['extension'])
    md5 = hashlib.md5(open(filepath, 'rb').read()).hexdigest()
    coordinate_space = entities.get('space', required_entities_with_defaults['coordinate_space'])
    hemisphere = entities.get('hemi', None)
    full_filepath = os.path.abspath(filepath)
    result = {
        'filetype': extension,
        'md5': md5,
        'coordinate_space_id': get_or_create_coordinate_space(session, coordinate_space).id,
        'subject_id': subject_obj.id,
        'path': full_filepath,
        'user_id': get_user_id(session)
    }
    if hemisphere:
        result['hemisphere'] = hemisphere
    return result

def process_connectivity_file(session, filepath, entities, metadata, subject_obj):
    result_dict = process_imaging_file(session, filepath, entities, metadata, subject_obj)
    
    # Fetch or create Connectome and StatisticType objects
    connectome_name = metadata.get('connectome', required_entities_with_defaults['connectome']).replace('u', '').replace('dil', '').lower()
    statistic_code = metadata.get('statistic', required_entities_with_defaults['statistic'])
    connectome_obj = get_or_create_connectome(session, name=connectome_name)
    statistic_type_obj = get_or_create_statistic_type(session, code=statistic_code)
    result_dict.update({
        'connectome_id': connectome_obj.id,
        'statistic_type_id': statistic_type_obj.id
    })

    insert_file_data_if_not_redundant(session, result_dict, ConnectivityFile)

def process_roi_file(session, filepath, entities, metadata, subject_obj):
    result_dict = process_imaging_file(session, filepath, entities, metadata, subject_obj)
    dimension = metadata.get('dimensionality', {}).get('predicted_value', 'unknown')
    dimension = session.query(Dimension).filter(func.lower(Dimension.name) == dimension.lower()).first()
    result_dict.update({
        'voxel_count': metadata.get('voxel_count', None),
        'dimension': dimension,
        'roi_type': metadata.get('mask_type', required_entities_with_defaults['roi_type'])
    })
    insert_file_data_if_not_redundant(session, result_dict, ROIFile)

def insert_file_data_if_not_redundant(session, result_dict, model_class):

    existing = session.query(model_class).filter_by(path=result_dict['path']).first()
    if existing:
        if existing.md5 == result_dict['md5']:
            print(f"Warning: File already exists with the same MD5 hash: {result_dict['md5']}. Skipping.")
        else:
            print(f"Warning: File path exists but with a different MD5 hash. Skipping.")
        return
    
    new_entry = model_class(**result_dict)
    session.add(new_entry)
    session.commit()