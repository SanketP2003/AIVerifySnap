# Build stage
FROM maven:3.9.6-eclipse-temurin-21-jammy AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the pom.xml and source code
COPY pom.xml .
COPY src ./src
# Also copy mvnw and .mvn if you prefer to use the wrapper
COPY mvnw .
COPY .mvn .mvn

# Build the application (skipping tests to speed up the build)
RUN mvn clean package -DskipTests

# Run stage
FROM eclipse-temurin:21-jre-jammy

# Set the working directory
WORKDIR /app

# Copy the built jar file from the build stage
# The jar file name usually depends on your pom.xml version/name, using wildcard to grab it
COPY --from=build /app/target/*.jar app.jar

# Render uses the PORT environment variable to bind the web service
ENV PORT=8080
EXPOSE 8080

# Run the jar file
ENTRYPOINT ["java", "-Dserver.port=${PORT}", "-jar", "app.jar"]
