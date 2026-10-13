package com.backend.aiverifysnap.controller;

import com.backend.aiverifysnap.model.DetectionHistory;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.repository.UserRepository;
import com.backend.aiverifysnap.service.DetectionService;
import com.backend.aiverifysnap.service.UserService;
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
    private final UserService userService;
    private final UserRepository userRepository;

    public DetectionController(DetectionService detectionService, UserService userService, UserRepository userRepository) {
        this.detectionService = detectionService;
        this.userService = userService;
        this.userRepository = userRepository;
    }

    public ResponseEntity<Map<String, Object>> validateImageFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty"));
        }
        if (file.getSize() > 10 * 1024 * 1024) {
            return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE)
                    .body(Map.of("error", "File size exceeds the 10MB maximum limit"));
        }
        String contentType = file.getContentType();
        String originalFilename = file.getOriginalFilename();
        boolean validExt = originalFilename != null && (
                originalFilename.toLowerCase().endsWith(".jpg") ||
                originalFilename.toLowerCase().endsWith(".jpeg") ||
                originalFilename.toLowerCase().endsWith(".png") ||
                originalFilename.toLowerCase().endsWith(".webp") ||
                originalFilename.toLowerCase().endsWith(".jfif"));
        boolean validMime = contentType != null && (
                contentType.equalsIgnoreCase("image/jpeg") ||
                contentType.equalsIgnoreCase("image/png") ||
                contentType.equalsIgnoreCase("image/webp") ||
                contentType.equalsIgnoreCase("application/octet-stream"));
        if (!validExt && !validMime) {
            return ResponseEntity.badRequest()
                    .body(Map.of("error", "Invalid file type. Only JPEG, PNG, and WebP images are supported."));
        }
        try {
            byte[] header = new byte[8];
            int read = file.getInputStream().read(header);
            if (read >= 2) {
                boolean isJpeg = (header[0] == (byte) 0xFF && header[1] == (byte) 0xD8);
                boolean isPng = read >= 4 && (header[0] == (byte) 0x89 && header[1] == (byte) 0x50 && header[2] == (byte) 0x4E && header[3] == (byte) 0x47);
                boolean isWebp = read >= 4 && (header[0] == (byte) 0x52 && header[1] == (byte) 0x49 && header[2] == (byte) 0x46 && header[3] == (byte) 0x46);
                if (!isJpeg && !isPng && !isWebp) {
                    return ResponseEntity.badRequest()
                            .body(Map.of("error", "Corrupted or unsupported image file header signature."));
                }
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("error", "Failed to inspect image file: " + e.getMessage()));
        }
        return null;
    }

    @PostMapping(value = "/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> analyzeImage(
            @Parameter(description = "Image file to analyze") @RequestParam("file") MultipartFile file,
            @Parameter(description = "User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        ResponseEntity<Map<String, Object>> validationError = validateImageFile(file);
        if (validationError != null) {
            return validationError;
        }

        if (userId != null) {
            Users user = userRepository.findById(userId).orElse(null);
            if (user != null) {
                String plan = user.getPlan() != null ? user.getPlan().toUpperCase() : "STARTER";
                if ("STARTER".equals(plan)) {
                    java.time.LocalDate today = java.time.LocalDate.now();
                    int scansToday = (user.getLastScanDate() != null && user.getLastScanDate().equals(today))
                            ? (user.getScansToday() != null ? user.getScansToday() : 0)
                            : 0;
                    if (scansToday >= 10) {
                        return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                                .body(Map.of(
                                        "error", "Daily quota limit reached",
                                        "message", "Starter accounts are limited to 10 scans per day. Upgrade to Professional for unlimited access."
                                ));
                    }
                }
            }
        }

        Map<String, Object> result = detectionService.detectDeepfake(file);
        if (result.containsKey("error")) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(result);
        }
        String imagePath = file.getOriginalFilename();
        detectionService.saveDetectionAsync(result, imagePath, userId);
        if (userId != null) {
            userService.recordUserScan(userId);
        }
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/analyze-bulk", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> analyzeBulk(
            @Parameter(description = "Image files to analyze (up to 50)") @RequestParam("files") MultipartFile[] files,
            @Parameter(description = "User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        if (files == null || files.length == 0) {
            return ResponseEntity.badRequest().body(Map.of("error", "No files provided"));
        }

        int maxBulk = 1;
        if (userId != null) {
            Users user = userRepository.findById(userId).orElse(null);
            if (user != null) {
                String plan = user.getPlan() != null ? user.getPlan().toUpperCase() : "STARTER";
                if ("PROFESSIONAL".equals(plan)) maxBulk = 50;
                else if ("ENTERPRISE".equals(plan)) maxBulk = 100;
            }
        }
        if (files.length > maxBulk) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body(Map.of(
                            "error", "Bulk scan limit exceeded",
                            "message", "Your current subscription tier permits up to " + maxBulk + " files per batch."
                    ));
        }

        for (MultipartFile file : files) {
            ResponseEntity<Map<String, Object>> validationError = validateImageFile(file);
            if (validationError != null) {
                return validationError;
            }
        }

        Map<String, Object> result = detectionService.detectBulkDeepfake(files, userId);
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/reverse-search", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> reverseSearch(
            @Parameter(description = "Image file for Google Lens reverse search") @RequestParam("file") MultipartFile file) {
        ResponseEntity<Map<String, Object>> validationError = validateImageFile(file);
        if (validationError != null) {
            return validationError;
        }
        Map<String, Object> result = detectionService.reverseImageSearch(file);
        return ResponseEntity.ok(result);
    }

    @PostMapping(value = "/identity-protect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, Object>> identityProtect(
            @Parameter(description = "User portrait reference image") @RequestParam("file") MultipartFile file,
            @Parameter(description = "User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        ResponseEntity<Map<String, Object>> validationError = validateImageFile(file);
        if (validationError != null) {
            return validationError;
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
            @Parameter(description = "Scan ID of the detection") @PathVariable Long scanId,
            @Parameter(description = "Requesting User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        DetectionHistory detection = detectionService.getDetectionById(scanId, userId);
        return ResponseEntity.ok(detection);
    }

    @DeleteMapping("/history/{scanId}")
    public ResponseEntity<Map<String, String>> deleteDetectionById(
            @Parameter(description = "Scan ID of the detection to delete") @PathVariable Long scanId,
            @Parameter(description = "Requesting User ID (optional)") @RequestParam(value = "userId", required = false) Long userId) {
        detectionService.deleteDetection(scanId, userId);
        return ResponseEntity.ok(Map.of("message", "Scan history deleted successfully", "scanId", scanId.toString()));
    }

    @GetMapping("/metrics")
    public ResponseEntity<Map<String, Object>> getMetrics() {
        List<DetectionHistory> all = detectionService.getAllDetections();
        long total = all.size();
        long fakes = all.stream().filter(h -> "Fake".equalsIgnoreCase(h.getResultLabel())).count();
        long reals = all.stream().filter(h -> "Real".equalsIgnoreCase(h.getResultLabel())).count();
        double avgConfidence = total > 0
                ? all.stream().mapToDouble(h -> h.getConfidenceScore() != null ? h.getConfidenceScore() : 0.0).average().orElse(0.0)
                : 0.0;
        return ResponseEntity.ok(Map.of(
                "totalScans", total,
                "totalFake", fakes,
                "totalReal", reals,
                "averageConfidence", Math.round(avgConfidence * 10.0) / 10.0
        ));
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> getHealth() {
        return ResponseEntity.ok(Map.of(
                "status", "UP",
                "service", "AIVerifySnap Backend API",
                "timestamp", java.time.LocalDateTime.now().toString()
        ));
    }
}
