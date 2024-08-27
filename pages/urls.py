# pages/urls.py

from django.urls import path

from .views.simple_views import HomePageView, AboutPageView, form_testing_view, form_testing_alternate_view, usage_page_view
from.views.dataset_views import all_datasets_view, dataset_details_view, build_csv_view

urlpatterns = [
    path("", HomePageView.as_view(), name="home"),
    path("usage/", usage_page_view, name="usage"),
    path("datasets/", all_datasets_view, name="datasets"),
    path("datasets/<int:dataset_id>/", dataset_details_view, name="dataset_details"),
    path("form_testing/", form_testing_view, name="form_testing"),
    path("form_testing_alternate/", form_testing_alternate_view, name="form_testing_alternate"),
    path("build_csv/<int:dataset_id>/", build_csv_view, name='build_csv'),
]
