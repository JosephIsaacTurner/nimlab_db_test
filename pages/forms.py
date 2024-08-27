# pages/forms.py

from django import forms
from .models import Subject, Cause, ConnectivityFile, Connectome, StatisticType, Handedness, Sex, ROIFile, Dimension, OriginalImageFile, ImageModality, Parcellation
from sqlalchemy_utils import db_utils
import hashlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
import nibabel as nib
from io import BytesIO
import gzip
import os
import tempfile
from sqlalchemy_utils.db_session import get_session

class SubjectFormMultiSubject(forms.ModelForm):
    """
    A formset for creating multiple subjects at once. Each subject will have one OriginalImage,
    one ROI, and one ConnectivityFile. This is useful for batch uploading.
    """

    class Meta:
        model = Subject
        fields = ['age', 'sex', 'handedness', 'cause']

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super(SubjectFormMultiSubject, self).__init__(*args, **kwargs)

        self.fields['cause'].queryset = Cause.objects.all()
        self.fields['cause'].label_from_instance = lambda obj: f"{obj.name} - {obj.description}"

        default_cause = Cause.objects.filter(name="ischemic stroke").first()
        if default_cause:
            self.fields['cause'].initial = default_cause.id
        
        default_handedness = Handedness.objects.filter(handedness="unknown").first()
        if default_handedness:
            self.fields['handedness'].initial = default_handedness.id

        default_sex = Sex.objects.filter(sex='unknown').first()
        if default_sex:
            self.fields['sex'].initial = default_sex

        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        # Set the prefix dynamically based on the form's index in the formset
        form_index = self.prefix.split('-')[1] if '-' in self.prefix else '0'

        self.connectivity_file_form = ConnectivityFileForm(
            data=kwargs.get('data'),
            files=kwargs.get('files'),
            prefix=f'{self.prefix}-connectivity'
        )
        self.roi_file_form = ROIFileForm(
            data=kwargs.get('data'),
            files=kwargs.get('files'),
            prefix=f'{self.prefix}-roi'
        )
        self.original_image_form = OriginalImageForm(
            data=kwargs.get('data'),
            files=kwargs.get('files'),
            prefix=f'{self.prefix}-original_image'
        )

    def is_valid(self):
        forms_valid = all([
            super(SubjectFormMultiSubject, self).is_valid(),
            self.connectivity_file_form.is_valid(),
            self.roi_file_form.is_valid(),
            self.original_image_form.is_valid()
        ])
        return forms_valid

    def save(self, commit=True):
        instance = super(SubjectFormMultiSubject, self).save(commit=False)
        
        # Ensure the user is set on the Subject instance
        if self.user:
            instance.user = self.user
        else:
            raise ValueError("User must be set before saving the Subject instance.")
        
        if commit:
            instance.save()

            # Save related forms
            self.connectivity_file_form.save(subject=instance, user=self.user)
            self.roi_file_form.save(subject=instance, user=self.user)
            self.original_image_form.save(subject=instance, user=self.user)
        
        return instance

class SubjectForm(forms.ModelForm):
    class Meta:
        model = Subject
        fields = ['age', 'sex', 'handedness', 'cause']  # Later we want 'case_report', 'patient_cohort'

    def __init__(self, *args, **kwargs):

        self.user = kwargs.pop('user', None)
        data = kwargs.get('data', None)
        files = kwargs.get('files', None)

        super(SubjectForm, self).__init__(*args, **kwargs)

        self.fields['cause'].queryset = Cause.objects.all()
        self.fields['cause'].label_from_instance = lambda obj: f"{obj.name} - {obj.description}"

        default_cause = Cause.objects.filter(name="ischemic stroke").first()
        if default_cause:
            self.fields['cause'].initial = default_cause.id
        
        default_handedness = Handedness.objects.filter(handedness="unknown").first()
        if default_handedness:
            self.fields['handedness'].initial = default_handedness.id

        default_sex = Sex.objects.filter(sex='unknown').first()
        if default_sex:
            self.fields['sex'].initial = default_sex

        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        # Initialize multiple file forms
        self.connectivity_file_forms = [ConnectivityFileForm(data=data, files=files, prefix=f'connectivity_{i}') for i in range(1)]
        self.roi_file_forms = [ROIFileForm(data=data, files=files, prefix=f'roi_{i}') for i in range(1)]
        self.original_image_forms = [OriginalImageForm(data=data, files=files, prefix=f'original_image_{i}') for i in range(1)]

    def is_valid(self):
        subject_form_valid = super(SubjectForm, self).is_valid()
        is_valid = all([
            subject_form_valid,
            all(form.is_valid() for form in self.connectivity_file_forms if form.has_changed()),
            all(form.is_valid() for form in self.roi_file_forms if form.has_changed()),
            all(form.is_valid() for form in self.original_image_forms if form.has_changed())
        ])

        print(f"Overall form validity: {is_valid}")

        return is_valid

    def save(self, commit=True):
        instance = super(SubjectForm, self).save(commit=False)
        if self.user:
            instance.user = self.user
        if commit:
            instance.save()
            for form in self.connectivity_file_forms:
                if form.has_changed() and form.is_valid():
                    form.save(subject=instance, user=self.user)
            for form in self.roi_file_forms:
                if form.has_changed() and form.is_valid():
                    form.save(subject=instance, user=self.user)
            for form in self.original_image_forms:
                if form.has_changed() and form.is_valid():
                    form.save(subject=instance, user=self.user)
        return instance

    def get_connectivity_file_forms(self):
        return self.connectivity_file_forms
    
    def get_roi_file_forms(self):
        return self.roi_file_forms
    
    def get_original_image_forms(self):
        return self.original_image_forms

class ConnectivityFileForm(forms.ModelForm):
    class Meta:
        model = ConnectivityFile
        fields = ['statistic_type', 'connectome', 'path']

    def __init__(self, *args, **kwargs):
        super(ConnectivityFileForm, self).__init__(*args, **kwargs)

        self.fields['connectome'].queryset = Connectome.objects.all()
        self.fields['connectome'].label_from_instance = lambda obj: f"{obj.name} - {obj.description}"

        default_connectome = Connectome.objects.filter(name="GSP1000MF").first()
        if default_connectome:
            self.fields['connectome'].initial = default_connectome.id

        self.fields['statistic_type'].queryset = StatisticType.objects.all()
        self.fields['statistic_type'].label_from_instance = lambda obj: f"{obj.name} - {obj.description}"

        default_statistic_type = StatisticType.objects.filter(name="student's t").first()
        if default_statistic_type:
            self.fields['statistic_type'].initial = default_statistic_type.id

        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        self.fields['path'].widget.attrs.update({'name': 'connectivity_file_path'})

    def save(self, commit=True, subject=None, user=None):        
        print('Saving connectivity file')
        instance = super(ConnectivityFileForm, self).save(commit=False)
        instance.subject = subject
        instance.filetype = db_utils.determine_filetype(instance.path.name)
        instance.md5 = hashlib.md5(instance.path.read()).hexdigest()
        instance.parcellation = Parcellation.objects.filter(name='3209c91v').first()
        if user:
            instance.user = user
        if commit:
            instance.save()

        if instance.filetype in ['nii', 'nii.gz']:
            session = get_session()           
            nifti_file = db_utils.fetch_from_s3(instance.path.name)
            print(nifti_file.shape)
            db_utils.data_to_parcelwise_arrays_table(
                parcellation = db_utils.fetch_atlas_3209c91v(),
                voxelwise_map = nifti_file,
                session = session,
                strategy = 'mean',
                map_type = 'connectivity',
                voxelwise_map_name = instance.path.name
            )
            session.close()

        return instance

class ROIFileForm(forms.ModelForm):
    class Meta:
        model = ROIFile
        fields = ['dimension', 'path']

    def __init__(self, *args, **kwargs):
        super(ROIFileForm, self).__init__(*args, **kwargs)

        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        default_dimension = Dimension.objects.filter(name="2d").first()
        if default_dimension:
            self.fields['dimension'].initial = default_dimension.id

        self.fields['path'].widget.attrs.update({'name': 'roi_file_path'})

    def save(self, commit=True, subject=None, user=None):
        print('Saving roi file')
        instance = super(ROIFileForm, self).save(commit=False)
        instance.subject = subject
        instance.filetype = db_utils.determine_filetype(instance.path.name)
        instance.md5 = hashlib.md5(instance.path.read()).hexdigest()
        instance.parcellation = Parcellation.objects.filter(name='3209c91v').first()
        if user:
            instance.user = user
        if commit:
            instance.save()

        if instance.filetype in ['nii', 'nii.gz']:
            session = get_session()           
            nifti_file = db_utils.fetch_from_s3(instance.path.name)
            print(nifti_file.shape)
            db_utils.data_to_parcelwise_arrays_table(
                parcellation = db_utils.fetch_atlas_3209c91v(),
                voxelwise_map = nifti_file,
                session = session,
                strategy = 'sum',
                map_type = 'roi',
                voxelwise_map_name = instance.path.name
            )
            session.close()

        return instance

class OriginalImageForm(forms.ModelForm):
    class Meta:
        model = OriginalImageFile
        fields = ['image_modality', 'path']

    def __init__(self, *args, **kwargs):
        super(OriginalImageForm, self).__init__(*args, **kwargs)

        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        default_modality = ImageModality.objects.filter(name="T1").first()
        if default_modality:
            self.fields['image_modality'].initial = default_modality.id

        self.fields['path'].widget.attrs.update({'name': 'original_image_path'})

    def save(self, commit=True, subject=None, user=None):
        print('Saving original image file')
        instance = super(OriginalImageForm, self).save(commit=False)
        instance.subject = subject
        if user:
            instance.user = user
        if commit:
            instance.save()
        return instance