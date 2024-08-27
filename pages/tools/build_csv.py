import pandas as pd
from sqlalchemy import text

def build_csv(session, research_paper_id, coordinate_space='2mm', statistic_code='avgR', connectome_name='GSP1000MF'):
    base_query = """
    SELECT DISTINCT ON (subjects.nickname)
        research_papers.nickname as dataset,
        COALESCE(patient_cohorts.name, 'default') as cohort,
        COALESCE(
            case_reports.doi, 
            CASE 
                WHEN case_reports.pubmed_id IS NOT NULL THEN CONCAT('PMID-', cast(case_reports.pubmed_id as varchar)) 
                ELSE NULL 
            END, 
            'not_provided'
        ) as citation,
        subjects.nickname as subject,
        roi_files.path as roi_2mm,
        connectivity_files.path as {statistic_code_alias},
        roi_files.voxel_count as voxel_count
    FROM
        research_papers
    JOIN
        subject_research_papers ON subject_research_papers.research_paper_id = research_papers.id
    JOIN
        subjects ON subjects.id = subject_research_papers.subject_id
    LEFT JOIN
        patient_cohorts ON patient_cohorts.id = subjects.patient_cohort_id
    JOIN
        roi_files ON roi_files.subject_id = subjects.id
    JOIN
        coordinate_spaces ON roi_files.coordinate_space_id = coordinate_spaces.id
    JOIN
        connectivity_files ON connectivity_files.subject_id = subjects.id
    JOIN
        statistic_types ON connectivity_files.statistic_type_id = statistic_types.id
    JOIN
        connectomes ON connectivity_files.connectome_id = connectomes.id
    LEFT JOIN 
        case_reports ON case_reports.id = subjects.case_report_id
    WHERE
        research_papers.id = :research_paper_id
        AND coordinate_spaces.name = :coordinate_space
        AND statistic_types.code = :statistic_code
        AND connectomes.name = :connectome_name
    ORDER BY subjects.nickname, cohort, roi_files.voxel_count, connectivity_files.path;
    """

    # Safely replace the statistic_code alias in the base query
    query = text(base_query.replace("{statistic_code_alias}", statistic_code))

    parameters = {
        'research_paper_id': research_paper_id,
        'coordinate_space': coordinate_space,
        'statistic_code': statistic_code,
        'connectome_name': connectome_name
    }
    
    results = session.execute(query, parameters).fetchall()
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['cohort', 'subject', 'voxel_count'], inplace=True)
    
    session.rollback()
    session.close()
    
    return results_df

def build_csv_all_connectomes(session, research_paper_id, coordinate_space='2mm', statistic_code='avgR'):
    base_query = """
    SELECT DISTINCT ON (subjects.nickname)
        research_papers.nickname as dataset,
        COALESCE(patient_cohorts.name, 'default') as cohort,
        COALESCE(
            case_reports.doi, 
            CASE 
                WHEN case_reports.pubmed_id IS NOT NULL THEN CONCAT('PMID-', cast(case_reports.pubmed_id as varchar)) 
                ELSE NULL 
            END, 
            'not_provided'
        ) as citation,
        subjects.nickname as subject,
        COALESCE(sexes.name, 'not_provided') as sex,
        COALESCE(causes.name, 'not_provided') as cause,
        COALESCE(CAST(subjects.age as varchar), 'not_provided') as age,
        roi_files_2mm.path as roi_2mm,
        roi_files_2mm.voxel_count as voxel_count,
        MAX(CASE WHEN connectomes.name = 'GSP1000MF' THEN connectivity_files.path END) AS GSP1000_MF_{statistic_code_alias},
        MAX(CASE WHEN connectomes.name = 'Yeo1000' THEN connectivity_files.path END) AS Yeo1000_{statistic_code_alias},
        MAX(CASE WHEN connectomes.name = 'GSP500M' THEN connectivity_files.path END) AS GSP500_M_{statistic_code_alias},
        MAX(CASE WHEN connectomes.name = 'GSP500F' THEN connectivity_files.path END) AS GSP500_F_{statistic_code_alias},
        MAX(CASE WHEN connectomes.name = 'HCP842' THEN connectivity_files.path END) AS HCP842_disconMapsTDIFiberMap
    FROM
        research_papers
    JOIN
        subject_research_papers ON subject_research_papers.research_paper_id = research_papers.id
    JOIN
        subjects ON subjects.id = subject_research_papers.subject_id
    LEFT JOIN
        patient_cohorts ON patient_cohorts.id = subjects.patient_cohort_id
    JOIN
        roi_files as roi_files_2mm ON roi_files_2mm.subject_id = subjects.id
    JOIN
        coordinate_spaces ON roi_files_2mm.coordinate_space_id = coordinate_spaces.id
    JOIN
        connectivity_files ON connectivity_files.subject_id = subjects.id
    JOIN
        statistic_types ON connectivity_files.statistic_type_id = statistic_types.id
    JOIN
        connectomes ON connectivity_files.connectome_id = connectomes.id
    LEFT JOIN 
        case_reports ON case_reports.id = subjects.case_report_id
    LEFT JOIN
        sexes ON sexes.id = subjects.sex_id
    LEFT JOIN
        causes on causes.id = subjects.cause_id
    WHERE
        research_papers.id = :research_paper_id
        AND coordinate_spaces.name = :coordinate_space
        AND statistic_types.code in (:statistic_code, 'disconMapsDisconnectedStreamlineMap')
        AND connectivity_files.filetype in ('.nii.gz', '.nii', '.trk.gz.tdi.nii.gz')
        AND roi_files_2mm.filetype in ('.nii.gz', '.nii')
    GROUP BY
        research_papers.nickname, patient_cohorts.name, case_reports.doi, case_reports.pubmed_id, subjects.nickname, roi_files_2mm.path, roi_files_2mm.voxel_count, sexes.name, causes.name, subjects.age
    ORDER BY 
        subjects.nickname, cohort, roi_files_2mm.voxel_count;
    """

    # Safely replace the statistic_code alias in the base query
    query = text(base_query.replace("{statistic_code_alias}", statistic_code))

    parameters = {
        'research_paper_id': research_paper_id,
        'coordinate_space': coordinate_space,
        'statistic_code': statistic_code
    }
    
    results = session.execute(query, parameters).fetchall()
    results_df = pd.DataFrame(results)
    
    session.rollback()
    session.close()
    
    return results_df