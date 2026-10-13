package com.backend.aiverifysnap;

import com.backend.aiverifysnap.controller.DetectionController;
import com.backend.aiverifysnap.model.DetectionHistory;
import com.backend.aiverifysnap.model.Users;
import com.backend.aiverifysnap.repository.DetectionRepository;
import com.backend.aiverifysnap.repository.UserRepository;
import com.backend.aiverifysnap.service.DetectionService;
import com.backend.aiverifysnap.service.UserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DetectionServiceSecurityTest {

    @Mock
    private DetectionRepository detectionRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private UserService userService;

    private DetectionService detectionService;
    private DetectionController detectionController;

    @BeforeEach
    void setUp() {
        WebClient.Builder builder = WebClient.builder();
        detectionService = new DetectionService(
                builder,
                "http://localhost:8000",
                30,
                detectionRepository,
                userRepository,
                userService
        );
        detectionController = new DetectionController(detectionService, userService, userRepository);
    }

    @Test
    @DisplayName("P0: SHA-256 calculation must produce genuine cryptographic hash of file bytes")
    void testGenuineSha256Calculation() throws Exception {
        byte[] content = "SAMPLE_ORIGINAL_IMAGE_BYTES_12345".getBytes(StandardCharsets.UTF_8);
        MockMultipartFile file = new MockMultipartFile("file", "test.jpg", "image/jpeg", content);

        String computedHash = detectionService.calculateSha256(file);

        // Compute expected hash manually
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] expectedHashBytes = digest.digest(content);
        StringBuilder sb = new StringBuilder();
        for (byte b : expectedHashBytes) {
            sb.append(String.format("%02x", b));
        }
        String expectedHash = sb.toString();

        assertNotNull(computedHash);
        assertEquals(expectedHash, computedHash);
        assertNotEquals("7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f", computedHash);
    }

    @Test
    @DisplayName("P0: Identity Protection scan must return truthful empty findings without fabricated matches")
    void testIdentityProtectionScanTruthfulResults() {
        byte[] content = new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, 0x00, 0x11, 0x22};
        MockMultipartFile file = new MockMultipartFile("file", "portrait.jpg", "image/jpeg", content);

        Map<String, Object> result = detectionService.identityProtectionScan(file, 100L);

        assertNotNull(result);
        assertEquals(0, result.get("total_infringements_found"));
        assertEquals("NOMINAL", result.get("threat_level"));
        assertEquals(0.0, result.get("infringement_risk_score"));
        assertFalse((Boolean) result.get("takedown_eligible"));
        assertEquals("", result.get("dmca_notice_template"));

        @SuppressWarnings("unchecked")
        List<Object> matches = (List<Object>) result.get("detected_matches");
        assertNotNull(matches);
        assertTrue(matches.isEmpty(), "Matches must be empty, never fabricated");
    }

    @Test
    @DisplayName("P0: Multi-tenant user data isolation in getUserDetections")
    void testUserDataIsolation() {
        Long authenticatedUserId = 42L;

        // When userId is provided, must only query records belonging to that user
        when(detectionRepository.findByUserIdOrderByScanTimestampDesc(authenticatedUserId))
                .thenReturn(Collections.emptyList());

        List<DetectionHistory> results = detectionService.getUserDetections(authenticatedUserId);
        assertNotNull(results);
        verify(detectionRepository).findByUserIdOrderByScanTimestampDesc(authenticatedUserId);
        verify(detectionRepository, never()).findAll();

        // When userId is null (unauthenticated), must return empty list and never dump database
        List<DetectionHistory> unauthenticatedResults = detectionService.getUserDetections(null);
        assertNotNull(unauthenticatedResults);
        assertTrue(unauthenticatedResults.isEmpty());
    }

    @Test
    @DisplayName("P0: Record ownership verification on getDetectionById and deleteDetection")
    void testRecordOwnershipVerification() {
        DetectionHistory record = new DetectionHistory();
        record.setScanId(999L);
        Users owner = new Users();
        owner.setId(10L);
        record.setUser(owner);

        when(detectionRepository.findById(999L)).thenReturn(Optional.of(record));

        // Unauthorized user attempts to access record -> 403 Forbidden
        ResponseStatusException ex1 = assertThrows(ResponseStatusException.class, () ->
                detectionService.getDetectionById(999L, 9999L)
        );
        assertEquals(HttpStatus.FORBIDDEN, ex1.getStatusCode());

        // Unauthorized user attempts to delete record -> 403 Forbidden
        ResponseStatusException ex2 = assertThrows(ResponseStatusException.class, () ->
                detectionService.deleteDetection(999L, 9999L)
        );
        assertEquals(HttpStatus.FORBIDDEN, ex2.getStatusCode());
    }

    @Test
    @DisplayName("P1: Upload validation rejects non-image files and invalid magic bytes")
    void testUploadValidationRejectsInvalidFiles() {
        // Case 1: Empty file
        MockMultipartFile emptyFile = new MockMultipartFile("file", "test.jpg", "image/jpeg", new byte[0]);
        ResponseEntity<Map<String, Object>> res1 = detectionController.validateImageFile(emptyFile);
        assertNotNull(res1);
        assertEquals(HttpStatus.BAD_REQUEST, res1.getStatusCode());

        // Case 2: Invalid MIME type
        MockMultipartFile textFile = new MockMultipartFile("file", "payload.exe", "text/plain", "dummy".getBytes());
        ResponseEntity<Map<String, Object>> res2 = detectionController.validateImageFile(textFile);
        assertNotNull(res2);
        assertEquals(HttpStatus.BAD_REQUEST, res2.getStatusCode());

        // Case 3: Fake MIME type with wrong magic bytes
        MockMultipartFile spoofedFile = new MockMultipartFile("file", "spoofed.jpg", "image/jpeg", "NOT_A_JPEG".getBytes());
        ResponseEntity<Map<String, Object>> res3 = detectionController.validateImageFile(spoofedFile);
        assertNotNull(res3);
        assertEquals(HttpStatus.BAD_REQUEST, res3.getStatusCode());
        assertTrue(res3.getBody().get("error").toString().contains("header signature"));
    }

    @Test
    @DisplayName("P1: Upload validation accepts valid JPEG, PNG, and WebP signatures")
    void testUploadValidationAcceptsValidSignatures() {
        // Valid JPEG header
        byte[] jpegBytes = new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01};
        MockMultipartFile jpeg = new MockMultipartFile("file", "valid.jpg", "image/jpeg", jpegBytes);
        assertNull(detectionController.validateImageFile(jpeg));

        // Valid PNG header
        byte[] pngBytes = new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D};
        MockMultipartFile png = new MockMultipartFile("file", "valid.png", "image/png", pngBytes);
        assertNull(detectionController.validateImageFile(png));

        // Valid WebP header: RIFF....WEBP
        byte[] webpBytes = new byte[]{0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50};
        MockMultipartFile webp = new MockMultipartFile("file", "valid.webp", "image/webp", webpBytes);
        assertNull(detectionController.validateImageFile(webp));
    }
}
