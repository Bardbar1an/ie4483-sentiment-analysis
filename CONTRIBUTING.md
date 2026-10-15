# Working on this project

The owner can invite teammates from the repository's **Settings > Collaborators > Add people** page. Each teammate should accept their invitation and work from their own GitHub account.

## Make a change

```sh
git clone https://github.com/Bardbar1an/ie4483-sentiment-analysis.git
cd ie4483-sentiment-analysis
git switch -c yourname/short-description
# Make and test your changes.
python -m unittest discover -s tests -v
git add path/to/changed-file
git commit -m "Describe the change"
git push -u origin yourname/short-description
```

Open a pull request to `main`, explain the change and checks, and ask a teammate to review it. Keep each pull request focused so the group can understand and discuss it.

## Contribution credit

Set Git's commit name and email to your own identity. Use an email associated with your GitHub account, or your GitHub-provided private commit email. GitHub attribution follows linked commit identities and contribution activity; adding a name to a document alone does not create commit credit.

Record actual roles and supporting commits or pull requests in `CONTRIBUTORS.md`. For work genuinely done together, a commit can include `Co-authored-by: Name <email>` trailers with the participating teammates' correct details. Keep the report's contribution statement consistent with the work performed.

## Before pushing

Do not commit course data, lecturer materials, models, review-level results or credentials. Keep local files in `data/course/` and `runs/`, which are ignored. Clear notebook outputs before staging a notebook:

```sh
python -m jupyter nbconvert --clear-output --inplace sentiment_course.ipynb
```

Check `git diff --cached` before pushing. Aggregate metrics and charts can go in `results/`; include enough settings to reproduce them. Do not tune against the already-inspected holdout and present the result as a fresh evaluation.
