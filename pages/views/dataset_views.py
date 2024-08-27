from django.http import HttpResponse
from pages.models import ResearchPaper
from django.shortcuts import render
from pages.models import ResearchPaper, Subject, GroupLevelMapFile
import pandas as pd

from ..tools.build_csv import build_csv, build_csv_all_connectomes
from sqlalchemy_utils.db_session import get_session

def all_datasets_view(request):
    papers = ResearchPaper.objects.all()
    return render(request, 'pages/all_datasets.html', {'papers': papers})

def dataset_details_view(request, dataset_id):
    paper = ResearchPaper.objects.get(pk=dataset_id)
    dataset_list = ResearchPaper.objects.values('title', 'id')
    dataset_list = [
        {'title': dataset['title'].replace('_', ' '), 'id': dataset['id']}
        for dataset in dataset_list
    ]
    subjects = build_csv_all_connectomes(session=get_session(), research_paper_id=dataset_id)
    subjects['is_pubmed'] = subjects['citation'].str.contains('PMID')
    subjects['is_doi'] = subjects['citation'].str.contains('/')
    multiple_cohorts = subjects['cohort'].nunique() > 1
    subjects = subjects.to_dict(orient='records')
    group_level_maps = GroupLevelMapFile.objects.filter(research_paper=dataset_id)
    return render(request, 'pages/dataset_details.html', {'paper': paper, 'group_level_maps': group_level_maps, 'subjects': subjects, 'multiple_cohorts': multiple_cohorts, 'dataset_list': dataset_list})

def build_csv_view(request, dataset_id):
    research_paper = ResearchPaper.objects.get(pk=dataset_id)
    title = research_paper.title
    session = get_session()
    df = build_csv_all_connectomes(session=session, research_paper_id=dataset_id)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{title}_dataset.csv"'
    df.to_csv(response, index=False)
    return response

