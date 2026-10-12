package com.backend.aiverifysnap.controller;

import com.backend.aiverifysnap.model.DetectionHistory;
import com.backend.aiverifysnap.service.DetectionService;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/detection")
@Tag(name = "Detection", description = "Deepfake detection API")
public class DetectionController {

    private final DetectionService detectionService;

    public DetectionController(DetectionService detectionService) {
        this.detectionService = detectionService;
    }

    @PostMapping(value = "/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> analyzeImage(
            @Parameter(description = "Image file to analyze") @RequestParam("file") MultipartFile file,
            @Parameter(description = "User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty"));
        }
        Map<String, Object> result = detectionService.detectDeepfake(file);
        if (result.containsKey("error")) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(result);
        }
        String imagePath = file.getOriginalFilename();
        detectionService.saveDetectionAsync(result, imagePath, userId);
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/analyze-bulk", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> analyzeBulk(
            @Parameter(description = "Image files to analyze (up to 50)") @RequestParam("files") MultipartFile[] files,
            @Parameter(description = "User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        if (files == null || files.length == 0) {
            return ResponseEntity.badRequest().body(Map.of("error", "No files provided"));
        }
        Map<String, Object> result = detectionService.detectBulkDeepfake(files, userId);
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/reverse-search", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> reverseSearch(
            @Parameter(description = "Image file for Google Lens reverse search") @RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty"));
        }
        Map<String, Object> result = detectionService.reverseImageSearch(file);
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/identity-protect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> identityProtect(
            @Parameter(description = "User portrait reference image") @RequestParam("file") MultipartFile file,
            @Parameter(description = "User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty"));
        }
        Map<String, Object> result = detectionService.identityProtectionScan(file, userId);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/history")
    public ResponseEntity<List<DetectionHistory>> getAllDetections(
            @Parameter(description = "Filter by User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        List<DetectionHistory> detections = detectionService.getUserDetections(userId);
        return ResponseEntity.ok(detections);
    }

    @GetMapping("/history/{scanId}")
    public ResponseEntity<DetectionHistory> getDetectionById(
            @Parameter(description = "Scan ID of the detection") @PathVariable Long scanId) {
        DetectionHistory detection = detectionService.getDetectionById(scanId);
        return ResponseEntity.ok(detection);
    }

    @DeleteMapping("/history/{scanId}")
    public ResponseEntity<Map<String, String>> deleteDetectionById(
            @Parameter(description = "Scan ID of the detection to delete") @PathVariable Long scanId) {
        detectionService.deleteDetection(scanId);
        return ResponseEntity.ok(Map.of("message", "Scan history deleted successfully", "scanId", scanId.toString()));
    }
}
