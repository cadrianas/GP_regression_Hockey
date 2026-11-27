"""Network-free checks for season boundaries, downloads, caching, and run failures."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import zipfile

import pandas as pd

from src.config import StudyConfig, add_config_arguments, config_from_args, sample_labels
from src.download import SeasonLinks, ensure_data, ensure_season, fetch, sha256
from src.pipeline import execute


def csv_data(year, quality=.1):
    return pd.DataFrame([dict(season=year, game_id=20001, period=3, time=2500,
        homeTeamGoals=0, awayTeamGoals=2, homeTeamCode='A', awayTeamCode='B', teamCode='A',
        xGoal=quality, homeEmptyNet=0, awayEmptyNet=0, homeSkatersOnIce=5,
        awaySkatersOnIce=5, shotDistance=20, shotType='WRIST', xCordAdjusted=70,
        yCordAdjusted=10)]).to_csv(index=False).encode()


class ConfigTests(unittest.TestCase):
    def test_default_matches_published_split(self):
        years=StudyConfig().seasons()
        self.assertEqual(len(years),11)
        self.assertEqual(sum(m['era']=='early' for m in years.values()),8)

    def test_future_evaluation_is_separate_from_training(self):
        c=StudyConfig(train_end=2024,evaluation_season=2025)
        m=list(c.seasons().values())
        self.assertEqual([x['year'] for x in m if x['role']=='validate'],[2025])
        self.assertTrue(all(x['year']<2025 for x in m if x['role']=='train'))

    def test_reject_overlap_and_empty_era(self):
        for kwargs in [dict(train_end=2024),dict(recent_start=2014),dict(recent_start=2024),dict(train_start=True)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): StudyConfig(**kwargs)

    def test_new_year_does_not_inherit_complete_status(self):
        parser=argparse.ArgumentParser();add_config_arguments(parser)
        c=config_from_args(parser.parse_args(['--evaluation-season','2026']))
        self.assertEqual(c.evaluation_status,'snapshot')
        self.assertEqual(c.train_end,2023)

    def test_labels_follow_actual_seasons_and_status(self):
        full=pd.DataFrame({'era':['early','recent','validate'],'season':['2018-19','2024-25','2025-26']})
        full.attrs['evaluation_status']='season-to-date'
        self.assertEqual(sample_labels(full)['validate'],'2025-26 (season-to-date)')


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.target=self.root/'shots_2025.csv'

    def fake_archive(self,year=2025,quality=.1,name='shots_2025.csv'):
        def download(url,destination,**kwargs):
            with zipfile.ZipFile(destination,'w') as z: z.writestr(name,csv_data(year,quality))
        return download

    def test_catalog_accepts_only_published_season_links(self):
        p=SeasonLinks();p.feed('<a href="https://peter-tanner.com/moneypuck/downloads/shots_2025.zip">season</a><a href="https://evil.test/shots_2026.zip">bad</a>')
        self.assertEqual(list(p.links),[2025])

    def test_download_and_cached_reuse(self):
        with patch('src.download.fetch',side_effect=self.fake_archive()) as request:
            first=ensure_season(2025,self.root,{2025:'https://example.test/season'})
            second=ensure_season(2025,self.root)
            self.assertEqual(request.call_count,1)
        self.assertEqual(first['sha256'],second['sha256'])
        self.assertEqual(first['downloaded_at_utc'],second['downloaded_at_utc'])

    def test_refresh_preserves_previous_snapshot(self):
        self.target.write_bytes(csv_data(2025));digest=sha256(self.target)
        with patch('src.download.fetch',side_effect=self.fake_archive(quality=.2)):
            ensure_season(2025,self.root,{2025:'https://example.test/season'},refresh=True)
        self.assertTrue((self.root/'snapshots/2025'/f'{digest}.csv').exists())
        self.assertNotEqual(digest,sha256(self.target))

    def test_invalid_refresh_leaves_existing_file_intact(self):
        self.target.write_bytes(csv_data(2025));original=self.target.read_bytes()
        with patch('src.download.fetch',side_effect=self.fake_archive(year=2024)):
            with self.assertRaises(ValueError): ensure_season(2025,self.root,{2025:'url'},refresh=True)
        self.assertEqual(self.target.read_bytes(),original)
        self.assertFalse(list(self.root.glob('.download-*')))

    def test_zip_paths_cannot_escape(self):
        with patch('src.download.fetch',side_effect=self.fake_archive(name='../shots_2025.csv')):
            with self.assertRaises(ValueError): ensure_season(2025,self.root,{2025:'url'})
        self.assertFalse(self.target.exists())

    def test_unknown_season_fails_without_guessing_url(self):
        with self.assertRaisesRegex(ValueError,'no published archive'): ensure_season(2025,self.root,{})

    def test_offline_missing_files_never_access_network(self):
        with patch('src.download.catalog_links',side_effect=AssertionError('network')):
            with self.assertRaises(FileNotFoundError): ensure_data(StudyConfig(),self.root,offline=True)

    def test_season_to_date_forces_evaluation_refresh_only(self):
        c=StudyConfig(train_start=2022,recent_start=2023,train_end=2023,evaluation_season=2025,evaluation_status='season-to-date')
        for path in c.seasons(self.root):path.touch()
        with patch('src.download.catalog_links',return_value={}),patch('src.download.ensure_season',return_value={}) as check:
            ensure_data(c,self.root)
        self.assertEqual([call.kwargs['refresh'] for call in check.call_args_list],[False,False,True])

    def test_retry_transient_http_error(self):
        response=BytesIO(b'ok')
        with patch('src.download.urlopen',side_effect=[HTTPError('url',503,'busy',{},None),response]) as request,patch('src.download.time.sleep'):
            fetch('https://example.test/data',self.root/'download')
        self.assertEqual(request.call_count,2)
        self.assertEqual((self.root/'download').read_bytes(),b'ok')

    def test_permanent_http_error_is_not_retried(self):
        with patch('src.download.urlopen',side_effect=HTTPError('url',403,'denied',{},None)) as request:
            with self.assertRaisesRegex(RuntimeError,'HTTP 403'):fetch('https://example.test/data',self.root/'download')
        self.assertEqual(request.call_count,1)

    def test_failed_run_records_failure_and_rejects_reuse(self):
        output=self.root/'run'
        with patch('src.pipeline.ensure_data',side_effect=ValueError('bad data')):
            with self.assertRaises(ValueError):execute(StudyConfig(),self.root,output)
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['status'],'failed')
        self.assertIn('bad data',(output/'run.log').read_text())
        with self.assertRaises(FileExistsError):execute(StudyConfig(),self.root,output)


if __name__=='__main__':unittest.main()
