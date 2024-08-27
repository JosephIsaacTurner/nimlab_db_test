# pages/models.py

from django.db import models
from django.utils.timezone import now
from accounts.models import CustomUser
from django.core.files.storage import default_storage

"""Helper Functions"""

def connectivity_file_path(instance, filename):
    # Get the subject ID
    subject_id = instance.subject.id if instance.subject else 'unknown'
    
    # Create the path
    return f'uploads/subjects/sub-{subject_id}/connectivity/{filename}'

"""Table Classes"""

class BaseModel(models.Model):
    insert_date = models.DateTimeField(default=now, null=False)

    class Meta:
        abstract = True

    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

class TestUpload(BaseModel):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')

    def __str__(self):
        return self.name

    class Meta:
        managed = True
        db_table = 'test_uploads'

class Parcellation(BaseModel):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255)
    path = models.CharField(max_length=255, null=True, blank=True)
    md5 = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name
    
    class Meta:
        managed = False
        db_table = 'parcellations'

class Parcel(BaseModel):
    value = models.FloatField()
    label = models.CharField(max_length=255, null=True, blank=True)
    parcellation = models.ForeignKey(Parcellation, related_name='parcels', on_delete=models.CASCADE)

    def __str__(self):
        return self.label if self.label else str(self.value)
    
    class Meta:
        managed = False
        db_table = 'parcels'

class VoxelwiseValue(BaseModel):
    mni152_x = models.IntegerField()
    mni152_y = models.IntegerField()
    mni152_z = models.IntegerField()
    parcel = models.ForeignKey(Parcel, related_name='voxelwise_values', on_delete=models.CASCADE)

    def __str__(self):
        return f"Voxel({self.mni152_x}, {self.mni152_y}, {self.mni152_z})"
    
    class Meta:
        managed = False
        db_table = 'voxelwise_values'

class ParcelwiseConnectivityValue(BaseModel):
    value = models.FloatField()
    connectivity_file = models.ForeignKey('ConnectivityFile', related_name='parcelwise_connectivity_values', on_delete=models.CASCADE)
    parcel = models.ForeignKey('Parcel', related_name='parcelwise_connectivity_values', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'parcelwise_connectivity_values'

class ParcelwiseGroupLevelMapValue(BaseModel):
    value = models.FloatField()
    group_level_map_file = models.ForeignKey('GroupLevelMapFile', related_name='parcelwise_group_level_map_values', on_delete=models.CASCADE)
    parcel = models.ForeignKey('Parcel', related_name='parcelwise_group_level_map_values', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'parcelwise_group_level_map_values'

class ParcelwiseROIValue(BaseModel):
    value = models.FloatField()
    roi_file = models.ForeignKey('ROIFile', related_name='parcelwise_roi_values', on_delete=models.CASCADE)
    parcel = models.ForeignKey('Parcel', related_name='parcelwise_roi_values', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'parcelwise_roi_values'

class ConnectivityFile(models.Model):
    filetype = models.CharField(max_length=255)
    path = models.FileField(upload_to='uploads/')
    md5 = models.CharField(max_length=255)
    subject = models.ForeignKey('Subject', related_name='connectivity_files', on_delete=models.CASCADE)
    parcellation = models.ForeignKey('Parcellation', related_name='connectivity_files', on_delete=models.CASCADE, null=True, blank=True)
    connectome = models.ForeignKey('Connectome', related_name='connectivity_files', on_delete=models.CASCADE)
    statistic_type = models.ForeignKey('StatisticType', related_name='connectivity_files', on_delete=models.CASCADE)
    coordinate_space = models.ForeignKey('CoordinateSpace', related_name='connectivity_files', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'connectivity_files'

class GroupLevelMapFile(models.Model):
    map_type = models.IntegerField()
    filetype = models.CharField(max_length=255)
    path = models.FileField(upload_to='uploads/')
    control_cohort = models.CharField(max_length=255, null=True, blank=True)
    threshold = models.FloatField(null=True, blank=True)
    parcellation = models.ForeignKey('Parcellation', related_name='group_level_map_files', on_delete=models.CASCADE, null=True, blank=True)
    research_paper = models.ForeignKey('ResearchPaper', related_name='group_level_map_files_set', on_delete=models.CASCADE)
    statistic_type = models.ForeignKey('StatisticType', related_name='group_level_map_files', on_delete=models.CASCADE)
    coordinate_space = models.ForeignKey('CoordinateSpace', related_name='group_level_map_files', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'group_level_map_files'

class ROIFile(models.Model):
    filetype = models.CharField(max_length=255)
    path = models.FileField(upload_to='uploads/')
    md5 = models.CharField(max_length=255)
    voxel_count = models.IntegerField(null=True, blank=True)
    roi_type = models.CharField(max_length=255, null=True, blank=True)
    parcellation = models.ForeignKey('Parcellation', related_name='roi_files', on_delete=models.CASCADE, null=True, blank=True)
    subject = models.ForeignKey('Subject', related_name='roi_files', on_delete=models.CASCADE)
    dimension = models.ForeignKey('Dimension', related_name='roi_files', on_delete=models.CASCADE)
    coordinate_space = models.ForeignKey('CoordinateSpace', related_name='roi_files', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'roi_files'

class CoordinateSpace(models.Model):
    name = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'coordinate_spaces'

class Dimension(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name
    
    class Meta:
        managed = False
        db_table = 'dimensions'

class StatisticType(models.Model):
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'statistic_types'

class Connectome(BaseModel):
    name = models.CharField(max_length=255)
    connectome_type = models.CharField(max_length=255, null=True, blank=True)
    description = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name
    
    class Meta:
        managed = False
        db_table = 'connectomes'

class Subject(models.Model):
    age = models.IntegerField(null=True, blank=True)
    nickname = models.CharField(max_length=255, null=True, blank=True)
    case_report = models.ForeignKey('CaseReport', related_name='subjects', on_delete=models.CASCADE, null=True, blank=True)
    patient_cohort = models.ForeignKey('PatientCohort', related_name='subjects', on_delete=models.CASCADE, null=True, blank=True)
    cause = models.ForeignKey('Cause', related_name='subjects', on_delete=models.CASCADE, null=True, blank=True)
    sex = models.ForeignKey('Sex', related_name='subjects', on_delete=models.CASCADE, null=True, blank=True)
    handedness = models.ForeignKey('Handedness', related_name='subjects', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'subjects'

class Sex(BaseModel):
    sex = models.CharField(max_length=255, null=False, blank=False)

    def __str__(self):
        return self.sex

    class Meta:
        managed = False
        db_table = 'sexes'

class Handedness(BaseModel):
    handedness = models.CharField(max_length=255)

    def __str__(self):
        return self.handedness

    class Meta:
        managed = False
        db_table = 'handedness'

class Cause(BaseModel):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name
    
    class Meta:
        managed = False
        db_table = 'causes'

class OriginalImageFile(BaseModel):
    path = models.FileField(upload_to='uploads/')
    subject = models.ForeignKey('Subject', related_name='original_image_files', on_delete=models.CASCADE)
    image_modality = models.ForeignKey('ImageModality', related_name='original_image_files', on_delete=models.CASCADE)

    def __str__(self):
        return self.path
    
    class Meta:
        managed = False
        db_table = 'original_image_files'

class ImageModality(BaseModel):
    name = models.CharField(max_length=255)
    description = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name
    
    class Meta:
        managed = False
        db_table = 'image_modalities'

class CaseReport(BaseModel):
    doi = models.CharField(max_length=255, null=True, blank=True)
    pubmed_id = models.IntegerField(null=True, blank=True)
    other_citation = models.CharField(max_length=255, null=True, blank=True)
    title = models.CharField(max_length=255, null=True, blank=True)
    first_author = models.CharField(max_length=255, null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    abstract = models.CharField(max_length=255, null=True, blank=True)
    path = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.title if self.title else f"CaseReport {self.id}"
    
    class Meta:
        managed = False
        db_table = 'case_reports'

class PatientCohort(BaseModel):
    name = models.CharField(max_length=255)
    doi = models.CharField(max_length=255, null=True, blank=True)
    pubmed_id = models.IntegerField(null=True, blank=True)
    other_citation = models.CharField(max_length=255, null=True, blank=True)
    source = models.CharField(max_length=255, null=True, blank=True)
    first_author = models.CharField(max_length=255, null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    description = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name
    
    class Meta:
        managed = False
        db_table = 'patient_cohorts'

class InclusionCriteria(BaseModel):
    is_case_study = models.BooleanField()
    is_english = models.BooleanField()
    is_relevant_symptoms = models.BooleanField()
    is_relevant_clinical_scores = models.BooleanField()
    is_full_text = models.BooleanField()
    is_temporally_linked = models.BooleanField()
    is_brain_scan = models.BooleanField()
    is_included = models.BooleanField()
    notes = models.TextField(null=True, blank=True)
    patient_cohort = models.ForeignKey('PatientCohort', related_name='inclusion_criteria', on_delete=models.CASCADE, null=True, blank=True)
    case_report = models.ForeignKey('CaseReport', related_name='inclusion_criteria', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'inclusion_criteria'

class SubjectsSymptoms(models.Model):
    subject = models.ForeignKey('Subject', on_delete=models.CASCADE)
    symptom = models.ForeignKey('Symptom', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'subjects_symptoms'

class Symptom(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    domain = models.ForeignKey('Domain', related_name='symptoms', on_delete=models.CASCADE)
    subdomain = models.ForeignKey('Subdomain', related_name='symptoms', on_delete=models.CASCADE, null=True, blank=True)
    subjects = models.ManyToManyField('Subject', through='SubjectsSymptoms', related_name='symptoms')
    research_papers = models.ManyToManyField('ResearchPaper', through='ResearchPapersSymptoms', related_name='symptoms_m2m')

    class Meta:
        managed = False
        db_table = 'symptoms'

class Synonym(BaseModel):
    name = models.CharField(max_length=255)
    symptom = models.ForeignKey('Symptom', related_name='synonyms', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'synonyms'

class MeshTerm(BaseModel):
    name = models.CharField(max_length=255)
    symptom = models.ForeignKey('Symptom', related_name='mesh_terms', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'mesh_terms'

class Domain(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = 'domains'

class Subdomain(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    domain = models.ForeignKey('Domain', related_name='subdomains', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'subdomains'

class SubjectResearchPaper(models.Model):
    research_paper = models.ForeignKey('ResearchPaper', on_delete=models.CASCADE)
    subject = models.ForeignKey('Subject', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'subject_research_papers'

class ResearchPaper(models.Model):
    doi = models.CharField(max_length=255, null=True, blank=True)
    pubmed_id = models.IntegerField(null=True, blank=True)
    other_citation = models.CharField(max_length=255, null=True, blank=True)
    title = models.CharField(max_length=255, null=True, blank=True)
    year = models.IntegerField(null=True, blank=True)
    abstract = models.TextField(null=True, blank=True)
    nickname = models.CharField(max_length=255, null=True, blank=True)
    comments = models.TextField(null=True, blank=True)
    first_author = models.ForeignKey('Author', related_name='first_authored_papers_set', on_delete=models.CASCADE)
    authors = models.ManyToManyField('Author', through='ResearchPaperAuthors', related_name='co_authored_papers_set')
    # group_level_map_files = models.ForeignKey('GroupLevelMapFile', related_name='research_papers_set', on_delete=models.CASCADE, null=True, blank=True)
    symptoms = models.ManyToManyField('Symptom', through='ResearchPapersSymptoms', related_name='research_papers_set')
    tags = models.ManyToManyField('Tag', related_name='research_papers')
    subjects = models.ManyToManyField('Subject', through='SubjectResearchPaper', related_name='research_papers')

    class Meta:
        managed = False
        db_table = 'research_papers'

    def get_author_names(self):
        return ', '.join([author.name for author in self.authors.all()])
    
    def get_title(self):
        return self.title.replace('_', ' ')

class ResearchPaperAuthors(models.Model):
    research_paper = models.ForeignKey('ResearchPaper', on_delete=models.CASCADE)
    author = models.ForeignKey('Author', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'research_paper_authors'

class Author(models.Model):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255, null=True, blank=True)
    institution = models.CharField(max_length=255, null=True, blank=True)
    # first_authored_papers = models.ForeignKey('ResearchPaper', related_name='first_author_set', on_delete=models.CASCADE)
    # co_authored_papers = models.ManyToManyField('ResearchPaper', through='ResearchPaperAuthors', related_name='co_authors_set')

    class Meta:
        managed = False
        db_table = 'authors'

class ResearchPapersSymptoms(models.Model):
    research_paper = models.ForeignKey('ResearchPaper', on_delete=models.CASCADE)
    symptom = models.ForeignKey('Symptom', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'research_papers_symptoms'

class ClinicalMeasure(BaseModel):
    metric_name = models.CharField(max_length=255)
    value = models.FloatField()
    unit = models.CharField(max_length=255, null=True, blank=True)
    timepoint = models.IntegerField(null=True, blank=True)
    subject = models.ForeignKey('Subject', related_name='clinical_measures', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'clinical_measures'

class Tag(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    research_paper = models.ForeignKey('ResearchPaper', related_name='tags_fk', on_delete=models.CASCADE)

    class Meta:
        managed = False
        db_table = 'tags'