from git import Repo

repo_url = "https://github.com/Navadeep206/Job-portal/tree/main/job-portal-mern"

local_path = "data/repositories/job-portal-mern"

Repo.clone_from(repo_url, local_path)

print("Repository cloned successfully")