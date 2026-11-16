"""P2-01N pure-stdlib adversarial local/OCI host admissions; ZERO Docker runs."""
import copy
import unittest
from unittest import mock
from experiments import p2_01n_candidate_host_readiness_no_fit as s

def worker():
    return {
        "schema":s.WORKER_SCHEMA,
        "original_frozen_P1_SHA256":s.P1_SHA,
        "actual_Linux_amd64_Python_3_12_16":True,
        "backend_entrypoint_identities_resolved_without_fallback":
            {name:24 for name in s.c.BACKENDS},
        "resolved_exact_original_P1_case_identifiers":72,
        "original_Cython_native_SHA256":
            s.guard.k.lock.NEW_NATIVE["production_pcc_step"],
        "original_isolated_Cython_observer_SHA256":
            s.guard.k.lock.NEW_NATIVE["isolated_pcc_step_live_research"],
        "real_OCI_worker_kernel_memory_limit_bytes":s.CGROUP_MEM,
        "real_OCI_worker_swap_limit_bytes":0,
        "real_OCI_worker_pids_limit":128,
        "real_OCI_worker_cgroup_cpu_quota_numerator":100000,
        "real_OCI_worker_cgroup_cpu_quota_denominator":100000,
        "real_OCI_worker_tmpfs_free_bytes":s.TMPFS,
        "OCI_root_overlay_free_disk_bytes_at_check":24*1024**3,
        "Docker_local_anonymous_volume_free_disk_bytes_at_check":21*1024**3,
        "Docker_temporary_tmpfs_free_bytes_at_check":s.TMPFS,
        "local_anonymous_Docker_volume_mount_verified_non_tmpfs":True,
        "volume_is_temporary_auto_removed_by_docker_run_rm":True,
        "real_OCI_worker_native_PCC_propagations_executed":0,
        "real_original_P1_fits_executed":0,
        "P2_execution_authorized":False,
    }

def info():
    return {"OSType":"linux","Architecture":"x86_64",
            "CgroupVersion":"2","MemTotal":12*1024**3,
            "ServerVersion":"29.0.0"}

def context():
    return [{"Endpoints":{"docker":{"Host":"unix:///var/run/docker.sock"}}}]

def archive_proof():
    return {
        "host_physical_RAM_bytes":16*1024**3,
        "work_directory_free_disk_bytes_at_check":32*1024**3,
        "original_OCI_archive_length_bytes":s.IMAGE_BYTES,
        "original_OCI_archive_SHA256":s.IMAGE_SHA,
        "original_OCI_archive_was_preexisting_and_unchanged":True
    }

def report():
    return {
        "schema":s.SCHEMA,"status":s.STATUS,
        "P1_scientific_sha256":s.P1_SHA,
        "exact_original_OCI_gzip_SHA256":s.IMAGE_SHA,
        "original_OCI_config_ID":s.IMAGE_ID,
        "observed_local_host":{
            **archive_proof(),**s.validate_docker_info(info(),context())
        },
        "exact_original_OCI_import_only_worker":worker(),
        "candidate_thresholds_P2_07C_not_approved":True,
        "a_suitable_host_has_not_been_selected_or_approved":True,
        "source_compiler_SDK_OS_build_closure_proven":False,
        "real_original_50k_checkpoint_resource_bounds_proven":False,
        "Docker_engine_root_volume_free_disk_proven":False,
        "Docker_local_volume_and_OCI_root_disk_capacity_observed":True,
        "host_identity_paths_serials_tokens_exported":False,
        **s.NO_GO,
    }

class OriginalActualHostNoFitContract(unittest.TestCase):
    def test_original_frozen_source_and_OCI_and_P1(self):
        self.assertTrue(s.m.verify_source_blobs())
        self.assertTrue(s.guard.source_bytes_locked())
        self.assertEqual(s.IMAGE_SHA,
            "008b189f5326036aa6de95cb189dd2de3984cd0d93641c8a91ec30e6414097c8")
        self.assertEqual(s.IMAGE_BYTES,290796771)

    def test_valid_local_docker_engine_and_genuine_VM_RAM(self):
        x=s.validate_docker_info(info(),context())
        self.assertEqual(x["engine_VM_total_memory_bytes"],12*1024**3)
        self.assertTrue(x["daemon_proven_local_nonremote"])
        self.assertTrue(s.local_docker_endpoint(
            "npipe:////./pipe/dockerDesktopLinuxEngine"))

    def test_disallow_remote_Docker_ssh_tcp_even_with_good_image(self):
        for endpoint in (
            "ssh://nas.example","tcp://127.0.0.1:2375",
            "http://dockerhost","unix://", "npipe://machine",
        ):
            with self.subTest(endpoint=endpoint),self.assertRaises(s.HostBlocked):
                s.local_docker_endpoint(endpoint)
        for override in ("ssh://user@nas","tcp://127.0.0.1:2375"):
            with self.subTest(override=override),self.assertRaises(s.HostBlocked):
                s.local_docker_endpoint("unix:///var/run/docker.sock",override)

    def test_bad_OSType_arch_cgroup_engine_and_ram_fail_closed(self):
        for field,value in (
            ("OSType","windows"),
            ("Architecture","aarch64"),
            ("CgroupVersion","1"),
            ("MemTotal",8*1024**3-1),
            ("ServerVersion",""),
        ):
            mutated={**info(),field:value}
            with self.subTest(field=field),self.assertRaises(s.HostBlocked):
                s.validate_docker_info(mutated,context())

    def test_preloaded_image_sha_and_architecture_mandatory(self):
        correct=[{"Id":s.IMAGE_ID,"Os":"linux","Architecture":"amd64"}]
        self.assertTrue(s.verify_image_inspect(correct))
        for field,value in (
            ("Id","sha256:"+"a"*64),
            ("Os","windows"),
            ("Architecture","arm64"),
        ):
            altered=[{**correct[0],field:value}]
            with self.subTest(field=field),self.assertRaises(s.HostBlocked):
                s.verify_image_inspect(altered)

    def test_original_worker_is_import_only_no_script_selects_fit(self):
        cmd=s.docker_args("pypcc-p201n-012345abcdef")
        for field in (
            "--network","none","--read-only","--user","65534:65534",
            "--memory","2g","--memory-swap","--pids-limit","128",
            "P201N_IMPORT_ONLY=1",
            "PYTHONPATH="+s.guard.k.CORRECT_PATH,
            "NUMBA_CACHE_DIR=/tmp/numba-cache",
            f"type=volume,destination={s.DOCKER_VOLUME_MOUNT},volume-nocopy",
            "experiments.p2_01n_candidate_host_readiness_no_fit",
        ):
            self.assertIn(field,cmd)
        self.assertEqual(cmd[-1],"--worker")
        self.assertNotIn("fit_predict"," ".join(cmd))
        self.assertNotIn("docker pull"," ".join(cmd))
        self.assertNotIn("docker load"," ".join(cmd))
        self.assertNotIn("UCI"," ".join(cmd))
        for name in ("invalid","pypcc-p207r-012345abcdef",
                     "pypcc-p201n-injection;rm"):
            with self.subTest(name=name),self.assertRaises(s.HostBlocked):
                s.docker_args(name)

    def test_good_import_only_exact_72_case_worker_receipt(self):
        self.assertTrue(s.validate_worker(worker()))
        receipt=report()
        self.assertIs(s.validate_report(receipt),receipt)

    def test_original_Cython_fallback_and_wrong_native_SHA_rejected(self):
        for field,value in (
            ("backend_entrypoint_identities_resolved_without_fallback",
                {name:24 for name in list(s.c.BACKENDS)[:2]}),
            ("original_Cython_native_SHA256","0"*64),
            ("original_isolated_Cython_observer_SHA256","f"*64),
            ("resolved_exact_original_P1_case_identifiers",71),
            ("real_OCI_worker_native_PCC_propagations_executed",1),
            ("real_original_P1_fits_executed",1),
            ("P2_execution_authorized",True),
        ):
            obj=worker()
            obj[field]=value
            with self.subTest(field=field),self.assertRaises(s.HostBlocked):
                s.validate_worker(obj)

    def test_actual_host_cgroup_and_tmpfs_quotas_required(self):
        for field,value in (
            ("real_OCI_worker_kernel_memory_limit_bytes",2*1024**3+1),
            ("real_OCI_worker_swap_limit_bytes",42),
            ("real_OCI_worker_pids_limit",0),
            ("real_OCI_worker_cgroup_cpu_quota_numerator",200000),
            ("real_OCI_worker_cgroup_cpu_quota_denominator",99999),
            ("real_OCI_worker_tmpfs_free_bytes",0),
            ("real_OCI_worker_tmpfs_free_bytes",s.TMPFS+1),
            ("OCI_root_overlay_free_disk_bytes_at_check",s.DISK_MIN-1),
            ("Docker_local_anonymous_volume_free_disk_bytes_at_check",s.DISK_MIN-1),
            ("Docker_temporary_tmpfs_free_bytes_at_check",s.TMPFS-1),
            ("local_anonymous_Docker_volume_mount_verified_non_tmpfs",False),
            ("volume_is_temporary_auto_removed_by_docker_run_rm",False),
        ):
            obj=worker()
            obj[field]=value
            with self.subTest(field=field),self.assertRaises(s.HostBlocked):
                s.validate_worker(obj)

    def test_no_formal_host_selection_from_candidate_capacity(self):
        r=report()
        for field,value in (
            ("candidate_thresholds_P2_07C_not_approved",False),
            ("a_suitable_host_has_not_been_selected_or_approved",False),
            ("source_compiler_SDK_OS_build_closure_proven",True),
            ("real_original_50k_checkpoint_resource_bounds_proven",True),
            ("Docker_engine_root_volume_free_disk_proven",True),
            ("Docker_local_volume_and_OCI_root_disk_capacity_observed",False),
            ("host_identity_paths_serials_tokens_exported",True),
            ("P2_01_final_execution_toolchain_lock_pass",True),
            ("P2_07_full_resources_gate_pass",True),
            ("P2_execution_preflight_sha256","a"*64),
            ("P2_execution_authorized",True),
            ("real_PCC_fits_executed",1),
        ):
            x={**r,field:value}
            with self.subTest(field=field),self.assertRaises(s.HostBlocked):
                s.validate_report(x)

    def test_host_and_Docker_VM_RAM_disk_all_independently_required(self):
        for field,value in (
            ("host_physical_RAM_bytes",s.RAM_MIN-1),
            ("work_directory_free_disk_bytes_at_check",s.DISK_MIN-1),
            ("engine_VM_total_memory_bytes",s.RAM_MIN-1),
            ("daemon_proven_local_nonremote",False),
        ):
            x=report()
            x["observed_local_host"][field]=value
            with self.subTest(field=field),self.assertRaises(s.HostBlocked):
                s.validate_report(x)

    def test_real_archive_absent_or_wrong_len_or_hash_rejected(self):
        with mock.patch("pathlib.Path.is_file",return_value=False):
            with self.assertRaises(s.HostBlocked):
                s.exact_original_archive("not-a-file")
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as d:
            path=Path(d)/"image.tar.gz"
            path.write_bytes(b"wrong-original-OCI")
            with self.assertRaises(s.HostBlocked):
                s.exact_original_archive(path)

    def test_correct_capacities_need_original_existing_archive(self):
        with mock.patch.object(s,"exact_original_archive",return_value=True):
            with mock.patch("pathlib.Path.is_dir",return_value=True):
                self.assertEqual(s.check_candidate_host_archive_capacity(
                    "archive","good",s.RAM_MIN,s.DISK_MIN
                )["original_OCI_archive_SHA256"],s.IMAGE_SHA)
                for ram,free in (
                    (s.RAM_MIN-1,s.DISK_MIN),
                    (s.RAM_MIN,s.DISK_MIN-1),
                ):
                    with self.assertRaises(s.HostBlocked):
                        s.check_candidate_host_archive_capacity("archive","good",ram,free)

    def test_cpu_quota_parser_rejects_unlimited_bad_inputs(self):
        self.assertEqual(s.parse_cpu_quota("100000 100000"),(100000,100000))
        for value in ("max 100000","100000","0 100000",
                      "100001 100000","-100 100000","abc foo"):
            with self.subTest(value=value),self.assertRaises(s.HostBlocked):
                s.parse_cpu_quota(value)


    def test_actual_docker_local_volume_mountinfo_and_original_overlay_disk(self):
        mountpoint=s.DOCKER_VOLUME_MOUNT
        sample=("51 42 0:50 / / rw,relatime - overlay overlay rw\\n"
                f"52 51 8:1 /var/lib/docker/volumes/probe/_data {mountpoint} rw,relatime - ext4 /dev/vda1 rw\\n"
                "53 51 0:51 / /tmp rw,nosuid,nodev - tmpfs tmpfs rw\\n")
        observed=s.checked_docker_storage_space(
            sample,19*1024**3,22*1024**3,s.TMPFS)
        self.assertTrue(observed["local_anonymous_Docker_volume_mount_verified_non_tmpfs"])
        self.assertEqual(observed["OCI_root_overlay_free_disk_bytes_at_check"],
                         19*1024**3)
        self.assertEqual(observed["Docker_local_anonymous_volume_free_disk_bytes_at_check"],
                         22*1024**3)

    def test_missing_fake_or_non_disk_volume_fails_closed(self):
        mountpoint=s.DOCKER_VOLUME_MOUNT
        invalid=(
            "",
            "51 42 0:50 / / rw - overlay overlay rw\\n",
            f"52 51 0:51 / {mountpoint} rw - tmpfs tmpfs rw\\n",
            f"52 51 0:51 / {mountpoint} rw - overlay overlay rw\\n",
            (f"52 51 8:1 / {mountpoint} rw - ext4 /dev/vda1 rw\\n"
             f"53 51 8:1 / {mountpoint} rw - ext4 /dev/vda1 rw\\n"),
        )
        for mountinfo in invalid:
            with self.subTest(mount=mountinfo[:30]),self.assertRaises(s.HostBlocked):
                s.checked_docker_storage_space(
                    mountinfo,s.DISK_MIN,s.DISK_MIN,s.TMPFS)

    def test_docker_storage_disk_min_is_distinct_from_local_host_workdir(self):
        mountinfo=f"52 51 8:1 / {s.DOCKER_VOLUME_MOUNT} rw - ext4 /dev/vda1 rw"
        for overlay,volume,tmp in (
            (s.DISK_MIN-1,s.DISK_MIN,s.TMPFS),
            (s.DISK_MIN,s.DISK_MIN-1,s.TMPFS),
            (s.DISK_MIN,s.DISK_MIN,0),
            (s.DISK_MIN,s.DISK_MIN,s.TMPFS+1),
        ):
            with self.subTest(overlay=overlay,volume=volume,tmp=tmp):
                with self.assertRaises(s.HostBlocked):
                    s.checked_docker_storage_space(mountinfo,overlay,volume,tmp)

    def test_receipt_canonical_sha_and_no_sensitive_device_fields(self):
        r=report()
        self.assertEqual(s.digest(r),s.digest(copy.deepcopy(r)))
        self.assertIsNone(r["P2_execution_preflight_sha256"])
        self.assertFalse(r["host_identity_paths_serials_tokens_exported"])
        self.assertFalse(any(word in s.canonical(r).decode("ascii").lower()
                            for word in ("username","hostname","mac_address",
                                         "filesystem_path","docker_root_dir")))

if __name__=="__main__":
    unittest.main()
