# pages/views.py

from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from ..forms import SubjectForm, SubjectFormMultiSubject
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.forms import formset_factory, modelformset_factory
from django.http import HttpResponse
from sqlalchemy_utils import db_utils
import hashlib
from pages.models import Parcellation, Subject
import os
import markdown

class HomePageView(TemplateView):
    template_name = "pages/home.html"

class AboutPageView(TemplateView):
    template_name = "pages/about.html"

@login_required
def form_testing_view(request):
    if request.method == 'POST':
        form = SubjectForm(data=request.POST, files=request.FILES, user=request.user)
        if form.is_valid():
            subject = form.save()
            return HttpResponse('Form submitted successfully!')
        else:
            print(form.errors)
            return HttpResponse('Form submission failed!')
    else:
        form = SubjectForm(user=request.user)

    context = {
        'form': form,
        'connectivity_file_forms': form.get_connectivity_file_forms(),
        'roi_file_forms': form.get_roi_file_forms(),
        'original_image_forms': form.get_original_image_forms(),
    }

    return render(request, 'pages/form_testing.html', context)

@login_required
def form_testing_alternate_view(request):
    """View for handling the SubjectFormMultiSubject formset."""

    # Define the formset factory outside the if-else block so it's available for both GET and POST requests
    SubjectFormMultiSubjectSet = modelformset_factory(
        Subject,
        form=SubjectFormMultiSubject,
        extra=2  # Define how many forms to show initially
    )

    if request.method == 'POST':
        formset = SubjectFormMultiSubjectSet(data=request.POST, files=request.FILES, queryset=Subject.objects.none())

        if formset.is_valid():
            # Loop through each form in the formset and save them individually
            for form in formset:
                # Pass the user to the form
                form.user = request.user
                subject = form.save()
                print("Saved subject:", subject)
                
            return HttpResponse('Formset submitted successfully!')
        else:
            print(formset.errors)
            return HttpResponse('Formset submission failed!')
    else:
        formset = SubjectFormMultiSubjectSet(queryset=Subject.objects.none())

    context = {
        'formset': formset,
    }

    return render(request, 'pages/form_testing_alternate.html', context)

def usage_page_view(request, **kwargs):
    context = {}
    # We need to make the path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    markdown_path = os.path.join(project_root, "../templates/pages/usage.md")
    markdown_path = os.path.abspath(markdown_path)
    with open(markdown_path, "r") as file:
        markdown_content = file.read()
    html_content = markdown.markdown(markdown_content)
    context["html_content"] = html_content
    return render(request, "pages/usage.html", context)
