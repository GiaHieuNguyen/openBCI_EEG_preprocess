# Git Setup and Push Guide for EEG Repository

This guide provides step-by-step instructions to initialize Git for this repository and push it to your GitHub repository.

## Prerequisites
- Ensure Git is installed on your system. If not, install it using your package manager (e.g., `sudo apt install git` on Ubuntu).
- Create a GitHub repository (e.g., via https://github.com/new). Note the repository URL (e.g., `https://github.com/yourusername/eeg-repo.git`).

## Step-by-Step Setup

1. **Navigate to the repository directory**:
   Open a terminal and go to the root of your project:
   ```
   cd /home/hieu/EEG
   ```

2. **Initialize Git repository**:
   Run the following command to initialize a new Git repository:
   ```
   git init
   ```

3. **Add all files to the staging area**:
   Add all files in the repository to Git:
   ```
   git add .
   ```

4. **Commit the files**:
   Create your first commit with a descriptive message:
   ```
   git commit -m "Initial commit: Add EEG project files"
   ```

5. **Set up the remote repository**:
   Link your local repository to your GitHub repository. Replace `<your-github-repo-url>` with your actual GitHub repository URL:
   ```
   git remote add origin <your-github-repo-url>
   ```
   Example:
   ```
   git remote add origin https://github.com/yourusername/eeg-repo.git
   ```

6. **Push to GitHub**:
   Push your commits to the main branch on GitHub:
   ```
   git push -u origin main
   ```
   - If your default branch is `master` instead of `main`, use `master` in the command.
   - The `-u` flag sets the upstream branch for future pushes.

## Additional Notes
- If you encounter authentication issues, ensure you have set up SSH keys or use a personal access token for HTTPS.
- For future commits, use `git add .`, `git commit -m "message"`, and `git push`.
- If you have a `.gitignore` file (which you do), ensure sensitive files are ignored.

## Troubleshooting
- If `git push` fails due to branch name mismatch, check your default branch on GitHub and adjust accordingly.
- Run `git status` to check the current state of your repository at any time.