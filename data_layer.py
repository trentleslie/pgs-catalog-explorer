from abc import ABC, abstractmethod
from typing import Optional, Tuple
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

BASE_URL = "https://www.pgscatalog.org/rest"
CACHE_TTL = timedelta(days=30)
FORCE_REFRESH_DAYS = 30
CACHE_VERSION = 4


class PGSDataSource(ABC):
    """Abstract interface for PGS data access.
    
    This abstraction allows swapping between REST API (development) 
    and DuckDB bulk files (production) without changing the UI code.
    """
    
    @abstractmethod
    def get_scores(self, filters: Optional[dict] = None) -> pd.DataFrame:
        """Get all scores with optional filtering."""
        pass
    
    @abstractmethod
    def get_score_details(self, pgs_id: str) -> dict:
        """Get detailed information for a single score."""
        pass
    
    @abstractmethod
    def get_traits(self) -> pd.DataFrame:
        """Get all traits."""
        pass
    
    @abstractmethod
    def get_trait_categories(self) -> pd.DataFrame:
        """Get trait categories."""
        pass
    
    @abstractmethod
    def get_publications(self) -> pd.DataFrame:
        """Get all publications."""
        pass
    
    @abstractmethod
    def get_publication_details(self, pgp_id: str) -> dict:
        """Get detailed information for a single publication."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self, pgs_id: Optional[str] = None) -> pd.DataFrame:
        """Get performance metrics, optionally filtered by score ID."""
        pass
    
    @abstractmethod
    def get_ancestry_categories(self) -> dict:
        """Get ancestry category definitions."""
        pass
    
    @abstractmethod
    def get_evaluation_summary(self) -> pd.DataFrame:
        """Get summary of evaluations per score (count and ancestry coverage)."""
        pass


class APIDataSource(PGSDataSource):
    """REST API implementation for PGS Catalog data access.
    
    Uses the PGS Catalog REST API with caching for efficient queries.
    Rate limited to 100 queries/minute by the API.
    """
    
    def __init__(self):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'PGS-Catalog-Explorer/1.0'
        })
    
    def _fetch_paginated(self, endpoint: str, params: Optional[dict] = None, progress_callback=None) -> Tuple[list, bool]:
        """Fetch all pages from a paginated endpoint following 'next' until null.
        
        Returns:
            Tuple of (results list, is_complete bool)
            is_complete is True if all expected results were fetched successfully
        """
        import time
        
        results = []
        url = f"{self.base_url}{endpoint}"
        if params is None:
            params = {}
        params['limit'] = 100
        
        page = 0
        total_count = None
        total_pages = None
        is_complete = True
        
        while url:
            page += 1
            
            if page > 1:
                time.sleep(0.6)
            
            if progress_callback:
                progress_callback(page, total_pages, len(results), total_count)
            
            response = None
            max_retries = 3
            retry_delays = [2, 4, 8]
            
            for attempt in range(max_retries + 1):
                try:
                    response = self.session.get(url, params=params, timeout=30)
                    
                    if response.status_code >= 500:
                        if attempt < max_retries:
                            delay = retry_delays[attempt]
                            print(f"[PAGINATION] Page {page}: HTTP {response.status_code} error, retry {attempt + 1}/{max_retries} in {delay}s")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"[PAGINATION] Page {page}: HTTP {response.status_code} error after {max_retries} retries, stopping")
                            is_complete = False
                            url = None
                            break
                    
                    if 400 <= response.status_code < 500:
                        print(f"[PAGINATION] Page {page}: HTTP {response.status_code} client error (not retrying)")
                        is_complete = False
                        url = None
                        break
                    
                    response.raise_for_status()
                    break
                    
                except requests.Timeout:
                    if attempt < max_retries:
                        delay = retry_delays[attempt]
                        print(f"[PAGINATION] Page {page}: Timeout, retry {attempt + 1}/{max_retries} in {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[PAGINATION] Page {page}: Timeout after {max_retries} retries, stopping")
                        is_complete = False
                        url = None
                        break
                        
                except requests.RequestException as e:
                    if attempt < max_retries:
                        delay = retry_delays[attempt]
                        print(f"[PAGINATION] Page {page}: Request error ({e}), retry {attempt + 1}/{max_retries} in {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[PAGINATION] Page {page}: Request error after {max_retries} retries: {e}")
                        is_complete = False
                        url = None
                        break
            
            if response is None or url is None:
                break
            
            try:
                data = response.json()
                
                if isinstance(data, dict) and 'results' in data:
                    page_results = len(data['results'])
                    results.extend(data['results'])
                    next_url = data.get('next')
                    print(f"[PAGINATION] Page {page}: fetched {page_results} results (total so far: {len(results)}), next={next_url is not None}")
                    url = next_url
                    params = {}
                    
                    if total_count is None:
                        total_count = data.get('count', 0)
                        if total_count > 0:
                            total_pages = (total_count + 99) // 100
                        print(f"[PAGINATION] API reports total count: {total_count}, expected pages: {total_pages}")
                else:
                    results = data if isinstance(data, list) else [data]
                    print(f"[PAGINATION] Page {page}: non-paginated response, {len(results)} results")
                    break
            except Exception as e:
                print(f"[PAGINATION] Page {page}: JSON parse error: {e}")
                is_complete = False
                break
        
        if total_count is not None and len(results) != total_count:
            is_complete = False
        
        status = "Complete" if is_complete else "INCOMPLETE"
        print(f"[PAGINATION] {status}: {page} pages fetched, {len(results)} total results (expected: {total_count})")
        
        if progress_callback:
            progress_callback(page, total_pages, len(results), total_count, complete=True)
        
        return results, is_complete
    
    def _fetch_single(self, endpoint: str) -> dict:
        """Fetch a single resource."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"API request failed: {e}")
            return {}
    
    def _get_api_count(self, endpoint: str) -> int:
        """Get total count from API endpoint with minimal data transfer (limit=1)."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params={'limit': 1}, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('count', 0) if isinstance(data, dict) else 0
        except requests.RequestException:
            return -1
    
    def get_api_counts(self) -> Tuple[int, int]:
        """Get current API counts for scores and evaluations."""
        scores_count = self._get_api_count('/score/all')
        evals_count = self._get_api_count('/performance/all')
        return scores_count, evals_count
    
    def check_cache_freshness(self, cached_scores_count: int, cached_evals_count: int) -> Tuple[bool, int, int]:
        """Check if cache is fresh by comparing counts with API.
        
        Returns: (is_fresh, api_scores_count, api_evals_count)
        """
        api_scores, api_evals = self.get_api_counts()
        if api_scores < 0 or api_evals < 0:
            return True, cached_scores_count, cached_evals_count
        is_fresh = (api_scores == cached_scores_count and api_evals == cached_evals_count)
        return is_fresh, api_scores, api_evals
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def get_scores(_self, filters: Optional[dict] = None, _version=CACHE_VERSION) -> Tuple[pd.DataFrame, bool]:
        """Get all scores with optional filtering.
        
        Returns:
            Tuple of (DataFrame, is_complete) where is_complete indicates
            whether all expected results were fetched successfully.
        """
        params = {}
        if filters:
            if filters.get('pgs_ids'):
                params['filter_ids'] = ','.join(filters['pgs_ids'])
        
        scores, is_complete = _self._fetch_paginated('/score/all', params)
        
        if not scores:
            return pd.DataFrame(), is_complete
        
        rows = []
        for score in scores:
            trait_names = []
            trait_ids = []
            has_efo = False
            has_mondo = False
            has_hp = False
            
            for trait in score.get('trait_efo', []):
                trait_names.append(trait.get('label', ''))
                trait_id = trait.get('id', '')
                trait_ids.append(trait_id)
                if trait_id.startswith('EFO_'):
                    has_efo = True
                elif trait_id.startswith('MONDO_'):
                    has_mondo = True
                elif trait_id.startswith('HP_'):
                    has_hp = True
            
            harmonized = score.get('ftp_harmonized_scoring_files', {})
            grch37_available = bool(harmonized.get('GRCh37', {}).get('positions'))
            grch38_available = bool(harmonized.get('GRCh38', {}).get('positions'))
            
            ancestry_dist = score.get('ancestry_distribution', {})
            dev_ancestry = []
            eval_ancestry = []
            if ancestry_dist:
                for stage, data in ancestry_dist.items():
                    if 'dev' in stage.lower() or 'gwas' in stage.lower():
                        dev_ancestry.extend(data.keys() if isinstance(data, dict) else [])
                    elif 'eval' in stage.lower():
                        eval_ancestry.extend(data.keys() if isinstance(data, dict) else [])
            
            pub = score.get('publication', {})
            
            has_any_ontology = has_efo or has_mondo or has_hp
            
            rows.append({
                'pgs_id': score.get('id', ''),
                'name': score.get('name', ''),
                'trait_names': '; '.join(trait_names),
                'trait_ids': '; '.join(trait_ids),
                'has_efo': has_efo,
                'has_mondo': has_mondo,
                'has_hp': has_hp,
                'has_any_ontology': has_any_ontology,
                'method_name': score.get('method_name', ''),
                'method_params': score.get('method_params', ''),
                'n_variants': score.get('variants_number', 0),
                'pgp_id': pub.get('id', ''),
                'publication_date': pub.get('date_publication', ''),
                'journal': pub.get('journal', ''),
                'first_author': pub.get('firstauthor', ''),
                'doi': pub.get('doi', ''),
                'ftp_scoring_file': score.get('ftp_scoring_file', ''),
                'grch37_available': grch37_available,
                'grch38_available': grch38_available,
                'grch37_url': harmonized.get('GRCh37', {}).get('positions', ''),
                'grch38_url': harmonized.get('GRCh38', {}).get('positions', ''),
                'dev_ancestry': '; '.join(set(dev_ancestry)),
                'eval_ancestry': '; '.join(set(eval_ancestry)),
                'weight_type': score.get('weight_type', ''),
                'license': score.get('license', ''),
                'date_release': score.get('date_release', ''),
            })
        
        return pd.DataFrame(rows), is_complete
    
    def get_score_details(self, pgs_id: str) -> dict:
        """Get detailed information for a single score."""
        return self._fetch_single(f'/score/{pgs_id}')
    
    def get_score_by_id(self, pgs_id: str) -> Optional[dict]:
        """Fetch a single score directly from API and parse into structured dict.
        
        Returns None if score not found, otherwise returns dict with fields:
        pgs_id, pgp_id, first_author, publication_date, trait_names, trait_efo,
        method_name, n_variants, grch37_available, grch38_available
        """
        raw = self._fetch_single(f'/score/{pgs_id}')
        if not raw or 'id' not in raw:
            return None
        
        pub = raw.get('publication', {}) or {}
        traits = raw.get('trait_efo', []) or []
        trait_names = '; '.join(t.get('label', '') for t in traits if isinstance(t, dict))
        trait_ids = '; '.join(t.get('id', '') for t in traits if isinstance(t, dict))
        
        ftp_grch37 = raw.get('ftp_harmonized_scoring_files', {}) or {}
        ftp_grch38 = ftp_grch37
        grch37 = bool(ftp_grch37.get('GRCh37', {}).get('positions'))
        grch38 = bool(ftp_grch38.get('GRCh38', {}).get('positions'))
        
        return {
            'pgs_id': raw.get('id', ''),
            'pgp_id': pub.get('id', ''),
            'first_author': pub.get('firstauthor', ''),
            'publication_date': pub.get('date_publication', ''),
            'doi': pub.get('doi', ''),
            'trait_names': trait_names,
            'trait_efo': trait_ids,
            'method_name': raw.get('method_name', ''),
            'n_variants': raw.get('variants_number', 0),
            'grch37_available': grch37,
            'grch38_available': grch38,
        }
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner="Loading traits...")
    def get_traits(_self) -> pd.DataFrame:
        """Get all traits."""
        traits, _ = _self._fetch_paginated('/trait/all')
        
        if not traits:
            return pd.DataFrame()
        
        rows = []
        for trait in traits:
            trait_cats = trait.get('trait_categories', [])
            cat_labels = []
            for c in trait_cats:
                if isinstance(c, dict):
                    cat_labels.append(c.get('label', ''))
                elif isinstance(c, str):
                    cat_labels.append(c)
            
            rows.append({
                'trait_id': trait.get('id', ''),
                'label': trait.get('label', ''),
                'description': trait.get('description', ''),
                'url': trait.get('url', ''),
                'associated_pgs_ids': '; '.join(trait.get('associated_pgs_ids', [])),
                'n_scores': len(trait.get('associated_pgs_ids', [])),
                'categories': '; '.join(cat_labels),
            })
        
        return pd.DataFrame(rows)
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner="Loading trait categories...")
    def get_trait_categories(_self) -> pd.DataFrame:
        """Get trait categories."""
        categories, _ = _self._fetch_paginated('/trait_category/all')
        
        if not categories:
            return pd.DataFrame()
        
        rows = []
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            efo_traits = cat.get('efotraits', [])
            trait_ids = []
            for t in efo_traits:
                if isinstance(t, dict):
                    trait_ids.append(t.get('id', ''))
                elif isinstance(t, str):
                    trait_ids.append(t)
            
            rows.append({
                'category': cat.get('label', ''),
                'n_traits': len(efo_traits),
                'trait_ids': '; '.join(trait_ids),
            })
        
        return pd.DataFrame(rows)
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner="Loading publications...")
    def get_publications(_self) -> pd.DataFrame:
        """Get all publications."""
        publications, _ = _self._fetch_paginated('/publication/all')
        
        if not publications:
            return pd.DataFrame()
        
        rows = []
        for pub in publications:
            associated = pub.get('associated_pgs_ids', {}) or {}
            dev_scores = [s for s in (associated.get('development', []) or []) if s]
            eval_scores = [s for s in (associated.get('evaluation', []) or []) if s]
            
            rows.append({
                'pgp_id': pub.get('id', ''),
                'title': pub.get('title', ''),
                'first_author': pub.get('firstauthor', ''),
                'journal': pub.get('journal', ''),
                'date_publication': pub.get('date_publication', ''),
                'doi': pub.get('doi', ''),
                'pmid': pub.get('PMID', ''),
                'n_development': len(dev_scores),
                'n_evaluation': len(eval_scores),
                'development_pgs_ids': '; '.join(dev_scores),
                'evaluation_pgs_ids': '; '.join(eval_scores),
                'date_release': pub.get('date_release', ''),
            })
        
        return pd.DataFrame(rows)
    
    def get_publication_details(self, pgp_id: str) -> dict:
        """Get detailed information for a single publication."""
        return self._fetch_single(f'/publication/{pgp_id}')
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner="Loading performance metrics...")
    def get_performance_metrics(_self, pgs_id: Optional[str] = None) -> pd.DataFrame:
        """Get performance metrics, optionally filtered by score ID.
        
        Returns DataFrame with individual metric columns:
        - auc, auc_ci: Area Under ROC Curve with confidence interval
        - r2, r2_ci: R-squared (variance explained)
        - or_val, or_ci: Odds Ratio with confidence interval
        - hr, hr_ci: Hazard Ratio with confidence interval
        - beta, beta_ci: Beta coefficient with confidence interval
        """
        if pgs_id:
            params = {'pgs_id': pgs_id}
            metrics, _ = _self._fetch_paginated('/performance/search', params)
        else:
            metrics, _ = _self._fetch_paginated('/performance/all')
        
        if not metrics:
            return pd.DataFrame()
        
        def extract_metric(metric_list, exact_names, partial_names=None):
            """Extract a specific metric by name.
            
            Args:
                metric_list: List of metric dicts to search
                exact_names: List of exact name_short values to match (case-insensitive)
                partial_names: Optional list of substrings for fallback matching
            """
            exact_lower = [n.lower() for n in exact_names]
            for m in metric_list:
                name_short = m.get('name_short', '').lower().strip()
                if name_short in exact_lower:
                    estimate = m.get('estimate')
                    ci_lower = m.get('ci_lower')
                    ci_upper = m.get('ci_upper')
                    ci_str = f"[{ci_lower}-{ci_upper}]" if ci_lower and ci_upper else None
                    return estimate, ci_str
            
            if partial_names:
                partial_lower = [n.lower() for n in partial_names]
                for m in metric_list:
                    name_long = m.get('name_long', '').lower()
                    if any(p in name_long for p in partial_lower):
                        estimate = m.get('estimate')
                        ci_lower = m.get('ci_lower')
                        ci_upper = m.get('ci_upper')
                        ci_str = f"[{ci_lower}-{ci_upper}]" if ci_lower and ci_upper else None
                        return estimate, ci_str
            return None, None
        
        rows = []
        for metric in metrics:
            pub = metric.get('publication', {})
            
            pm = metric.get('performance_metrics', {})
            effect_sizes = pm.get('effect_sizes', [])
            class_acc = pm.get('class_acc', [])
            other_metrics = pm.get('othermetrics', [])
            all_raw = effect_sizes + class_acc + other_metrics
            
            all_metrics = []
            for m in all_raw:
                name = m.get('name_short', m.get('name_long', ''))
                estimate = m.get('estimate', '')
                ci_lower = m.get('ci_lower', '')
                ci_upper = m.get('ci_upper', '')
                if estimate:
                    if ci_lower and ci_upper:
                        all_metrics.append(f"{name}: {estimate} [{ci_lower}-{ci_upper}]")
                    else:
                        all_metrics.append(f"{name}: {estimate}")
            
            auc, auc_ci = extract_metric(all_raw, ['auc', 'auroc', 'c-statistic', 'c-index'], ['area under'])
            r2, r2_ci = extract_metric(all_raw, ['r²', 'r2'], ['r-squared', 'variance explained'])
            or_val, or_ci = extract_metric(all_raw, ['or'], ['odds ratio'])
            hr, hr_ci = extract_metric(all_raw, ['hr'], ['hazard ratio'])
            beta, beta_ci = extract_metric(all_raw, ['beta', 'β'], None)
            
            samples = metric.get('sampleset', {}).get('samples', [])
            sample_size = 0
            ancestries = []
            cohorts = []
            for sample in samples:
                sample_size += sample.get('sample_number', 0) or 0
                ancestry = sample.get('ancestry_broad', '')
                if ancestry:
                    ancestries.append(ancestry)
                for cohort in sample.get('cohorts', []):
                    cohorts.append(cohort.get('name_short', cohort.get('name_full', '')))
            
            rows.append({
                'ppm_id': metric.get('id', ''),
                'pgs_id': metric.get('associated_pgs_id', ''),
                'pgp_id': pub.get('id', ''),
                'first_author': pub.get('firstauthor', ''),
                'publication_date': pub.get('date_publication', ''),
                'doi': pub.get('doi', ''),
                'phenotyping_reported': metric.get('phenotyping_reported', ''),
                'covariates': metric.get('covariates', ''),
                'sample_size': sample_size,
                'ancestry': '; '.join(set(ancestries)),
                'cohorts': '; '.join(set(cohorts)),
                'auc': auc,
                'auc_ci': auc_ci,
                'r2': r2,
                'r2_ci': r2_ci,
                'or_val': or_val,
                'or_ci': or_ci,
                'hr': hr,
                'hr_ci': hr_ci,
                'beta': beta,
                'beta_ci': beta_ci,
                'metrics': '; '.join(all_metrics),
                'effect_sizes': effect_sizes,
                'class_acc': class_acc,
                'other_metrics': other_metrics,
            })
        
        return pd.DataFrame(rows)
    
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def get_evaluation_summary(_self, _version=CACHE_VERSION) -> pd.DataFrame:
        """Get summary of evaluations per score (count and ancestry coverage).
        
        Follows 'next' until null to get all evaluations.
        
        Returns a DataFrame with:
        - pgs_id: Score ID
        - n_evaluations: Number of evaluations
        - n_ancestry_groups: Number of unique ancestry groups evaluated
        - ancestry_groups: Semicolon-separated list of ancestry groups
        """
        results, _ = _self._fetch_paginated('/performance/all')
        
        if not results:
            return pd.DataFrame()
        
        score_evals = {}
        
        for metric in results:
            pgs_id = metric.get('associated_pgs_id', '')
            if not pgs_id:
                continue
            
            if pgs_id not in score_evals:
                score_evals[pgs_id] = {
                    'n_evaluations': 0,
                    'ancestry_groups': set()
                }
            
            score_evals[pgs_id]['n_evaluations'] += 1
            
            samples = metric.get('sampleset', {}).get('samples', [])
            for sample in samples:
                ancestry = sample.get('ancestry_broad', '')
                if ancestry:
                    score_evals[pgs_id]['ancestry_groups'].add(ancestry)
        
        rows = []
        for pgs_id, data in score_evals.items():
            rows.append({
                'pgs_id': pgs_id,
                'n_evaluations': data['n_evaluations'],
                'n_ancestry_groups': len(data['ancestry_groups']),
                'ancestry_groups': '; '.join(sorted(data['ancestry_groups']))
            })
        
        return pd.DataFrame(rows)
    
    @st.cache_data(ttl=CACHE_TTL)
    def get_ancestry_categories(_self) -> dict:
        """Get ancestry category definitions."""
        return _self._fetch_single('/ancestry_categories')


def get_data_source() -> PGSDataSource:
    """Factory function to get the configured data source.
    
    Currently returns APIDataSource. In production, this could be
    configured to return DuckDBDataSource for better performance.
    """
    return APIDataSource()
