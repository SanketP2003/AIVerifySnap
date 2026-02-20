package com.backend.aiverifysnap.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.time.Duration;
import java.util.Map;

@RestController
@RequestMapping("/api/detection")
@Tag(name = "Detection", description = "Deepfake detection API")
public class DetectionController {

    private final WebClient webClient;
    private final int timeout;

    public DetectionController(
            WebClient.Builder webClientBuilder,
            @Value("${ml.service.url}") String mlServiceUrl,
            @Value("${ml.service.timeout:30}") int timeout) {
        this.webClient = webClientBuilder.baseUrl(mlServiceUrl).build();
        this.timeout = timeout;
    }

    @Operation(
            summary = "Detect deepfake in image",
            description = "Upload an image file to analyze for AI-generated or manipulated content"
    )
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Detection successful"),
            @ApiResponse(responseCode = "400", description = "Invalid image file"),
            @ApiResponse(responseCode = "500", description = "Detection service error")
    })
    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map> detectDeepfake(
            @Parameter(description = "Image file to analyze (JPEG, PNG, etc.)")
            @RequestParam("file") MultipartFile file) throws IOException {

        if (file.isEmpty()) {
            throw new IllegalArgumentException("File cannot be empty");
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            throw new IllegalArgumentException("Invalid file type. Please upload an image file.");
        }

        byte[] fileBytes = file.getBytes();
        String filename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "image";

        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", new ByteArrayResource(fileBytes) {
            @Override
            public String getFilename() {
                return filename;
            }
        }).contentType(MediaType.parseMediaType(contentType));

        Map response = webClient.post()
                .uri("/predict")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(timeout))
                .block();

        return ResponseEntity.ok(response);
    }

    @Operation(
            summary = "Detect deepfake in base64 image",
            description = "Submit a base64-encoded image for deepfake detection"
    )
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Detection successful"),
            @ApiResponse(responseCode = "400", description = "Invalid base64 image"),
            @ApiResponse(responseCode = "500", description = "Detection service error")
    })
    @PostMapping("/predict-base64")
    public ResponseEntity<Map> detectDeepfakeBase64(@RequestBody Map<String, String> payload) {

        String imageBase64 = payload.get("image_base64");
        if (imageBase64 == null || imageBase64.isEmpty()) {
            throw new IllegalArgumentException("image_base64 field is required");
        }

        Map response = webClient.post()
                .uri("/predict-base64")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of("image_base64", imageBase64))
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(timeout))
                .block();

        return ResponseEntity.ok(response);
    }

    @Operation(
            summary = "Check ML service health",
            description = "Check if the ML detection service is running and accessible"
    )
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Health check result")
    })
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> checkHealth() {
        try {
            webClient.get()
                    .uri("/health")
                    .retrieve()
                    .bodyToMono(String.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();
            return ResponseEntity.ok(Map.of("ml_service_healthy", true, "status", "ok"));
        } catch (Exception e) {
            return ResponseEntity.ok(Map.of("ml_service_healthy", false, "status", "unavailable"));
        }
    }
}

