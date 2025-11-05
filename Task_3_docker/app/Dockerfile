# Use a standard Python 3.9 base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application files (code, ref_data, etc.) into the container
COPY . .

# Define the command to run when the container starts
# It will execute your main prediction script
CMD ["python", "predict.py", "--input_csv", "/app/data/input.csv", "--output_json", "/app/out/output.json"]